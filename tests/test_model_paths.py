"""Run with python3 -m unittest discover -s tests -v (no ML runtime needed)."""
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import typing
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
import AILab_model_paths as paths


def node_definitions(filename):
    """Execute actual node classes/schemas with ML dependencies replaced.

    Imports and top-level hardware initialization are excluded, so these tests
    exercise node interfaces and loader dispatch on a machine without ComfyUI.
    """
    tree = ast.parse((ROOT / "py" / filename).read_text())
    namespace = dict(vars(typing))
    namespace.update(vars(paths))
    namespace.update({"__name__": filename.removesuffix(".py"), "__file__": str(ROOT / "py" / filename),
        "os": os, "json": json, "Path": Path, "sys": sys,
        "device": "cpu", "DEVICE": "cpu", "torch": MagicMock(),
        "transforms": MagicMock(), "Image": MagicMock(), "np": MagicMock(),
        "PreTrainedModel": type("PreTrainedModel", (), {}),
        "release_model": MagicMock(), "clean_vram": MagicMock(),
        "get_device": lambda: "cpu", "F": MagicMock(), "comfy": MagicMock(),
        "importlib": __import__("importlib"), "types": types,
        "contextmanager": contextmanager,
    })
    body = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = value
    if "MODE_CONFIGS" in namespace:
        namespace["MODE_CHOICES"] = tuple(namespace["MODE_CONFIGS"])
    exec(compile(ast.Module(body=body, type_ignores=[]), filename, "exec"), namespace)
    return namespace


NODES = {
    "RMBG": ("RMBG",), "BiRefNet": ("BiRefNetRMBG",),
    "BodySegment": ("BodySegment",), "ClothSegment": ("ClothesSegment",),
    "FaceSegment": ("FaceSegment",), "FashionSegment": ("FashionSegmentClothing",),
    "LamaRemover": ("AILab_LamaRemover",), "FinegrainSegment": ("AILab_FinegrainBoxSegment",),
    "FeyNoBg": ("AILab_FeyNoBg",), "Florence2": ("AILab_Florence2",),
    "VRMBG": ("AILab_VRMBG",), "SDMatte": ("AILab_SDMatte",),
    "SAM3Segment": ("SAM3Segment",), "Segment": ("Segment",),
    "SegmentV2": ("SegmentV2",), "SAM2Segment": ("SAM2Segment",),
    "Yolo": ("AILab_Yolo",), "YoloV8": ("AILab_YoloV8", "AILab_YoloV8Adv"),
}


class ModelPathsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.weight = self.root / "arbitrary model name.safetensors"
        self.weight.write_bytes(b"test weights")

    def test_validate_paths_before_loading(self):
        for value in (None, "", "  ", "relative/model.pt"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "model_path"):
                paths.model_file(value)
        for value in (str(self.root), str(self.root / "missing.pt")):
            with self.subTest(value=value), self.assertRaisesRegex(FileNotFoundError, "model_path"):
                paths.model_file(value)
        self.assertEqual(paths.model_file(f"  {self.weight}  "), str(self.weight.resolve()))
        alias = self.root / "alias.safetensors"
        alias.symlink_to(self.weight)
        self.assertEqual(paths.model_file(str(alias)), str(self.weight.resolve()))
        with patch.dict(os.environ, {"HOME": str(self.root)}):
            self.assertEqual(paths.model_file("~/" + self.weight.name), str(self.weight.resolve()))

    def test_model_and_temporal_cache_invalidation(self):
        @paths.local_model_inputs("model_path")
        class Node:
            FUNCTION = "run"

            def __init__(self):
                self.model = None
                self.temporal = None

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"image": ("IMAGE",)}}

            def run(self, **class_selections):
                self.asserted_inputs = class_selections
                if self.model is None:
                    self.model = object()
                self.temporal = self.model
                return self.model

        node = Node()
        first = node.run(model_path=str(self.weight), Hair=True)
        self.assertEqual(node.asserted_inputs, {"Hair": True})
        self.assertIs(node.run(model_path=str(self.weight)), first)
        second_path = self.root / "second.pt"
        second_path.write_bytes(b"second")
        self.assertIsNot(node.run(model_path=str(second_path)), first)
        with self.assertRaises(ValueError):
            node.run()

    def test_bundle_uses_repo_resources_and_preserves_user_files(self):
        resources = self.root / "resources" / "org" / "model"
        resources.mkdir(parents=True)
        (resources / "config.json").write_text('{"bundled": true}')
        (resources / "model.py").write_text("original = True\n")
        with patch.object(paths, "RESOURCE_ROOT", self.root / "resources"):
            bundle = paths.model_bundle("org/model", {"model.safetensors": str(self.weight)})
            self.assertEqual(Path(bundle, "model.safetensors").resolve(), self.weight.resolve())
            self.assertEqual(Path(bundle, "config.json").read_text(), '{"bundled": true}')
            Path(bundle, "model.py").write_text("adapted = True\n")
            self.assertEqual((resources / "model.py").read_text(), "original = True\n")
            self.assertEqual(self.weight.read_bytes(), b"test weights")
            self.assertEqual(paths.model_bundle("org/model", {"model.safetensors": str(self.weight)}), bundle)
            other = self.root / "other.pt"
            other.write_bytes(b"other")
            self.assertNotEqual(paths.model_bundle("org/model", {"model.safetensors": str(other)}), bundle)

    def test_all_node_schemas_have_required_paths(self):
        for module, classes in NODES.items():
            namespace = node_definitions(f"AILab_{module}.py")
            for name in classes:
                with self.subTest(node=name):
                    cls = namespace[name]
                    schema = cls.INPUT_TYPES()
                    expected = ("sam2_model_path", "dino_model_path", "bert_model_path") if module == "SAM2Segment" else (
                        ("sam_model_path", "dino_model_path", "bert_model_path") if module in ("Segment", "SegmentV2") else ("model_path",))
                    for field in expected:
                        self.assertEqual(schema["required"][field][0], "STRING")
                        self.assertEqual(schema["required"][field][1]["default"], "")

    def test_bundle_without_symlink_privileges(self):
        resources = self.root / "resources" / "org" / "windows"
        resources.mkdir(parents=True)
        (resources / "config.json").write_text("{}")
        with patch.object(paths, "RESOURCE_ROOT", self.root / "resources"), patch.object(Path, "symlink_to", side_effect=OSError("permission denied")):
            bundle = paths.model_bundle("org/windows", {"model.safetensors": str(self.weight)})
        copied = Path(bundle, "model.safetensors")
        self.assertFalse(copied.is_symlink())
        self.assertEqual(copied.read_bytes(), self.weight.read_bytes())
        copied.write_bytes(b"changed temporary copy")
        self.assertEqual(self.weight.read_bytes(), b"test weights")

    def test_all_decorated_nodes_reject_invalid_paths_before_execution(self):
        for module, classes in NODES.items():
            namespace = node_definitions(f"AILab_{module}.py")
            for name in classes:
                cls = namespace[name]
                if not hasattr(cls, "MODEL_PATH_FIELDS"):
                    continue
                # Bypass ML imports inside a few legacy constructors.
                instance = cls.__new__(cls)
                instance.model = None  # FashionSegment's destructor expects this.
                with self.subTest(node=name), self.assertRaisesRegex(ValueError, "_path"):
                    getattr(instance, cls.FUNCTION)()

    def test_weight_dispatch_for_directory_loaders(self):
        for module, cls_name, variant, filename in (
            ("RMBG", "BaseModelLoader", "RMBG-2.0", "model.safetensors"),
            ("RMBG", "BaseModelLoader", "BEN", "BEN_Base.pth"),
            ("RMBG", "BaseModelLoader", "BEN2", "BEN2_Base.pth"),
            ("BiRefNet", "BiRefNetModel", "BiRefNet-general", "BiRefNet-general.safetensors"),
            ("BiRefNet", "BiRefNetModel", "BiRefNet_lite", "BiRefNet_lite.safetensors"),
        ):
            with self.subTest(variant=variant):
                namespace = node_definitions(f"AILab_{module}.py")
                loader = namespace[cls_name]()
                loader.model_path = str(self.weight)
                bundle = loader.get_cache_dir(variant)
                self.assertEqual(Path(bundle, filename).resolve(), self.weight.resolve())
                self.assertTrue(Path(bundle, "BiRefNet_config.py" if module == "BiRefNet" or variant == "RMBG-2.0" else "model.py" if variant == "BEN" else "BEN2.py").is_file())

    def test_yolo_paths_and_simple_node_forwarding(self):
        for module, name in (("Yolo", "AILab_Yolo"), ("YoloV8", "AILab_YoloV8Adv")):
            namespace = node_definitions(f"AILab_{module}.py")
            node = namespace[name]()
            self.assertEqual(node._resolve_model_path(str(self.weight)), str(self.weight.resolve()))
            with self.assertRaises(ValueError):
                node._resolve_model_path("")
        namespace = node_definitions("AILab_YoloV8.py")
        with patch.object(namespace["AILab_YoloV8Adv"], "yolo_detect", return_value="result") as detect:
            result = namespace["AILab_YoloV8"]().yolo_detect_simple("images", str(self.weight))
            self.assertEqual(result, "result")
            self.assertEqual(detect.call_args.kwargs["model_path"], str(self.weight))

    def test_groundingdino_bert_is_local(self):
        args = types.SimpleNamespace(text_encoder_type="bert-base-uncased")
        result = paths.groundingdino_args(args, str(self.weight))
        self.assertIs(result, args)
        self.assertTrue(Path(args.text_encoder_type, "config.json").is_file())
        self.assertEqual(Path(args.text_encoder_type, "model.safetensors").resolve(), self.weight.resolve())

    def test_single_file_loader_dispatch(self):
        namespace = node_definitions("AILab_LamaRemover.py")
        node = namespace["AILab_LamaRemover"]()
        node.model_path = str(self.weight)
        node.load_model()
        namespace["torch"].jit.load.assert_called_once_with(str(self.weight), map_location="cpu")

        namespace = node_definitions("AILab_FinegrainSegment.py")
        solutions = types.ModuleType("refiners.solutions")
        solutions.BoxSegmenter = MagicMock()
        with patch.dict(sys.modules, {"refiners": types.ModuleType("refiners"), "refiners.solutions": solutions}):
            node = namespace["AILab_FinegrainBoxSegment"]()
            node.model_path = str(self.weight)
            node._ensure_model(margin=0.2)
            self.assertEqual(solutions.BoxSegmenter.call_args.kwargs["weights"], str(self.weight))

        namespace = node_definitions("AILab_RMBG.py")
        transparent = types.ModuleType("transparent_background")
        transparent.Remover = MagicMock()
        with patch.dict(sys.modules, {"transparent_background": transparent}):
            node = namespace["InspyrenetModel"]()
            node.model_path = str(self.weight)
            node.load_model("INSPYRENET")
            self.assertEqual(transparent.Remover.call_args.kwargs["ckpt"], str(self.weight))

    def test_onnx_and_segformer_dispatch(self):
        class DispatchReached(Exception):
            pass

        runtime = types.ModuleType("onnxruntime")
        runtime.InferenceSession = MagicMock(side_effect=DispatchReached("dispatch reached"))
        namespace = node_definitions("AILab_BodySegment.py")
        with patch.dict(sys.modules, {"onnxruntime": runtime}):
            with self.assertRaisesRegex(RuntimeError, "dispatch reached"):
                namespace["BodySegment"]().segment_body([], model_path=str(self.weight))
        runtime.InferenceSession.assert_called_once_with(str(self.weight.resolve()))

        for module, name in (("ClothSegment", "ClothesSegment"), ("FaceSegment", "FaceSegment"), ("FashionSegment", "FashionSegmentClothing")):
            with self.subTest(node=name):
                namespace = node_definitions(f"AILab_{module}.py")
                transformers = types.ModuleType("transformers")
                transformers.SegformerImageProcessor = MagicMock()
                transformers.AutoModelForSemanticSegmentation = MagicMock()
                transformers.AutoModelForSemanticSegmentation.from_pretrained.side_effect = DispatchReached("dispatch reached")
                node = namespace[name]()
                with patch.dict(sys.modules, {"transformers": transformers}), patch("builtins.print"):
                    with self.assertRaisesRegex(RuntimeError, "dispatch reached"):
                        getattr(node, node.FUNCTION)([], model_path=str(self.weight))
                call = transformers.AutoModelForSemanticSegmentation.from_pretrained.call_args
                self.assertTrue(call.kwargs["local_files_only"])
                self.assertEqual(Path(call.args[0], "model.safetensors").resolve(), self.weight.resolve())
                self.assertIsNone(node.processor)

    def test_florence_cache_includes_weight_file_and_attention(self):
        namespace = node_definitions("AILab_Florence2.py")
        namespace.update({"mm": MagicMock(), "Florence2Config": MagicMock(),
                          "Florence2": MagicMock(), "Processor": MagicMock(),
                          "load_torch_file": MagicMock(return_value={})})
        node = namespace["AILab_Florence2"]()
        node.model_path = str(self.weight)
        node._get_model("microsoft/Florence-2-base", "fp32", "sdpa")
        namespace["load_torch_file"].assert_called_once_with(str(self.weight))
        other = self.root / "another.bin"
        other.write_bytes(b"other")
        node.model_path = str(other)
        node._get_model("microsoft/Florence-2-base", "fp32", "sdpa")
        node._get_model("microsoft/Florence-2-base", "fp32", "eager")
        self.assertEqual(namespace["load_torch_file"].call_count, 3)
        self.assertEqual(len(node.MODEL_CACHE), 3)

    def test_sdmatte_load_failure_preserves_checkpoint(self):
        namespace = node_definitions("AILab_SDMatte.py")
        namespace.update({"SAFETENSORS_AVAILABLE": True,
                          "load_file": MagicMock(side_effect=ValueError("invalid weights"))})
        node = namespace["AILab_SDMatte"]()
        with self.assertRaisesRegex(RuntimeError, "invalid weights"):
            node._load_model_weights(MagicMock(), str(self.weight))
        self.assertEqual(self.weight.read_bytes(), b"test weights")

    def test_vrmbg_missing_gated_resources_is_actionable(self):
        namespace = node_definitions("AILab_VRMBG.py")
        node = namespace["AILab_VRMBG"]()
        node.model_path = str(self.weight)
        # Simulate an installation without approved access even if the developer
        # later populates the gated resource directory.
        directory = self.root / "resources" / "briaai" / "VRMBG-3.0"
        directory.mkdir(parents=True)
        (directory / "sources.json").write_text(json.dumps({"requires_access": ["model.py"]}))
        with patch.object(paths, "RESOURCE_ROOT", self.root / "resources"):
            with self.assertRaisesRegex(FileNotFoundError, "HF_TOKEN"):
                node._ensure_model()

    def test_workflows_have_empty_path_widgets(self):
        schemas = {}
        for module, names in NODES.items():
            namespace = node_definitions(f"AILab_{module}.py")
            for name in names:
                schemas[name] = namespace[name].INPUT_TYPES()
        for workflow in (ROOT / "example_workflows").glob("*.json"):
            for node in json.loads(workflow.read_text()).get("nodes", []):
                if node["type"] not in schemas:
                    continue
                index = 0
                for field, spec in schemas[node["type"]]["required"].items():
                    if not isinstance(spec[0], (list, tuple)) and spec[0] not in ("STRING", "INT", "FLOAT", "BOOLEAN", "COLORCODE"):
                        continue
                    if field.endswith("_path"):
                        with self.subTest(workflow=workflow.name, node=node["type"], field=field):
                            self.assertEqual(node["widgets_values"][index], "")
                    index += 1

    def test_vendored_files_match_pinned_hashes(self):
        for manifest_path in (ROOT / "models" / "resources").rglob("sources.json"):
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(manifest["revision"]), 40)
            for entry in manifest["files"]:
                with self.subTest(file=entry["file"], repo=manifest["repo"]):
                    self.assertIn(manifest["revision"], entry["url"])
                    self.assertEqual(hashlib.sha256((manifest_path.parent / entry["file"]).read_bytes()).hexdigest(), entry["sha256"])

    def test_no_download_or_delete_calls_in_node_loaders(self):
        forbidden = {"hf_hub_download", "snapshot_download", "download_url_to_file", "get_or_download_model_file", "remove", "unlink"}
        for module in NODES:
            tree = ast.parse((ROOT / "py" / f"AILab_{module}.py").read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                    with self.subTest(module=module, call=name):
                        self.assertNotIn(name, forbidden)


if __name__ == "__main__":
    unittest.main()

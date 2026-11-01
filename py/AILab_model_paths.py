"""Explicit local weight inputs and read-only, repository-owned model resources.

This module deliberately depends only on the standard library so validation can
run before any model is allocated. It never downloads files.
"""
import atexit
import json
from functools import wraps
from pathlib import Path
import shutil
import tempfile
import threading

RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "models" / "resources"
_bundles = {}
_bundle_lock = threading.RLock()


def model_file(value, field="model_path"):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: enter an absolute path to a local model weight file.")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field}: an absolute file path is required: {value!r}")
    if not path.is_file():
        raise FileNotFoundError(f"{field}: model file does not exist or is a directory: {path}")
    return str(path.resolve())


def checkpoint_state_dict(path):
    """Read one explicit checkpoint; never probe alternate/sibling files."""
    if Path(path).suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path)
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def resource_dir(repo):
    path = RESOURCE_ROOT / repo
    manifest = path / "sources.json"
    missing = []
    if manifest.is_file():
        missing = [name for name in json.loads(manifest.read_text()).get("requires_access", [])
                   if not (path / name).is_file()]
    if missing or not path.is_dir() or not any(p.name != "sources.json" for p in path.iterdir()):
        raise FileNotFoundError(
            f"Bundled model resources missing for {repo}: {path}. {', '.join(missing)} "
            "Restore the repository resources or run scripts/vendor_model_resources.py "
            "during setup (gated repositories require HF_TOKEN and approved access)."
        )
    return str(path)


def resource_file(repo, filename):
    path = Path(resource_dir(repo)) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Bundled resource missing: {path}")
    return str(path)


def model_bundle(repo, weights):
    """Expose explicit weights under names required by directory-based loaders.

    Weights are linked where supported, otherwise copied (e.g. Windows without
    symlink privileges). Original files are never renamed or written. A bundle
    lives until process exit, including lazy imports by custom models.
    """
    weights = tuple(sorted((name, model_file(path, name)) for name, path in weights.items()))
    key = (repo, weights)
    with _bundle_lock:
        if key not in _bundles:
            source = resource_dir(repo)
            temporary = tempfile.TemporaryDirectory(prefix="rmbg-model-")
            try:
                shutil.copytree(source, temporary.name, dirs_exist_ok=True)
                for name, path in weights:
                    destination = Path(temporary.name) / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        destination.symlink_to(path)
                    except (OSError, NotImplementedError):
                        shutil.copyfile(path, destination)
            except Exception:
                temporary.cleanup()
                raise
            _bundles[key] = temporary
        return _bundles[key].name


@atexit.register
def _cleanup_bundles():
    for temporary in _bundles.values():
        temporary.cleanup()
    _bundles.clear()


def local_model_inputs(*fields):
    """Add mandatory STRING weight inputs and invalidate per-instance caches.

    Path values are consumed here, before legacy **class_selections/**params
    handlers see them. Loaders use the normalized attributes of the same names.
    Each node retains its established execution arguments and output contract.
    """
    def decorate(cls):
        original_inputs = cls.INPUT_TYPES.__func__
        original_execute = getattr(cls, cls.FUNCTION)

        @classmethod
        def input_types(node_cls):
            schema = original_inputs(node_cls)
            required = schema.setdefault("required", {})
            for field in fields:
                required[field] = ("STRING", {
                    "default": "",
                    "tooltip": f"Absolute path to the local {field.removesuffix('_path')} weight file on the ComfyUI server. No automatic download.",
                })
            return schema

        @wraps(original_execute)
        def execute(self, *args, **kwargs):
            paths = {field: model_file(kwargs.pop(field, None), field) for field in fields}
            key = tuple(paths.items())
            if getattr(self, "_local_model_paths", None) != key:
                if hasattr(self, "_local_model_paths"):
                    # Release every instance cache, including temporal video state.
                    self.__dict__.clear()
                    self.__init__()
                self._local_model_paths = key
            for field, path in paths.items():
                setattr(self, field, path)
            return original_execute(self, *args, **kwargs)

        cls.INPUT_TYPES = input_types
        setattr(cls, cls.FUNCTION, execute)
        cls.MODEL_PATH_FIELDS = fields
        return cls
    return decorate


def groundingdino_args(args, bert_model_path):
    """Keep GroundingDINO's BERT tokenizer and encoder entirely local."""
    args.text_encoder_type = model_bundle("google-bert/bert-base-uncased", {
        "model.safetensors": bert_model_path,
    })
    return args

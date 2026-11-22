"""Test scaffolding: make the node modules importable without a full ComfyUI install.

The nodes import `folder_paths`, `nodes`, `comfy.utils`, `comfy.model_management` and
`comfy_extras.nodes_mask` from the ComfyUI host process. Those modules are absent when
the tests run standalone, so the shims below provide the same surface. When the real
modules are importable they are used unchanged.
"""
import importlib.util
import os
import sys
import tempfile
import types

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO_ROOT, "py") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "py"))


def _missing(name):
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ModuleNotFoundError):
        return True


def _install(name, module):
    sys.modules[name] = module


def _build_folder_paths():
    mod = types.ModuleType("folder_paths")
    base = tempfile.mkdtemp(prefix="rmbg_test_")
    mod.models_dir = os.path.join(base, "models")
    os.makedirs(mod.models_dir, exist_ok=True)

    def _dir(*parts):
        path = os.path.join(base, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    mod.get_input_directory = lambda: _dir("input")
    mod.get_temp_directory = lambda: _dir("temp")
    mod.get_output_directory = lambda: _dir("output")
    mod.get_annotated_filepath = lambda name, default_dir=None: os.path.join(_dir("input"), name)
    mod.exists_annotated_filepath = lambda name: os.path.isfile(mod.get_annotated_filepath(name))
    mod.get_full_path = lambda folder_name, filename: (
        os.path.join(mod.models_dir, folder_name, filename)
        if os.path.isfile(os.path.join(mod.models_dir, folder_name, filename)) else None
    )
    mod.get_filename_list = lambda folder_name: []
    mod.get_save_image_path = lambda prefix, output_dir, *a, **k: (output_dir, prefix, 1, "", "")
    return mod


def _build_nodes():
    mod = types.ModuleType("nodes")
    mod.MAX_RESOLUTION = 16384
    return mod


def _build_comfy_package():
    pkg = types.ModuleType("comfy")
    pkg.__path__ = []

    utils = types.ModuleType("comfy.utils")

    def common_upscale(samples, width, height, upscale_method, crop):
        """Same contract as ComfyUI's comfy/utils.py for the [B, C, H, W] case."""
        old_width = samples.shape[-1]
        old_height = samples.shape[-2]
        if crop == "center":
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            samples = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
        if samples.shape[-2] == height and samples.shape[-1] == width:
            return samples
        mode = "bicubic" if upscale_method == "lanczos" else upscale_method
        return torch.nn.functional.interpolate(samples, size=(height, width), mode=mode)

    class _ProgressBar:
        def __init__(self, total):
            self.total = total

        def update(self, value):
            pass

    utils.common_upscale = common_upscale
    utils.ProgressBar = _ProgressBar

    model_management = types.ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: torch.device("cpu")
    model_management.soft_empty_cache = lambda: None
    model_management.intermediate_device = lambda: torch.device("cpu")
    model_management.intermediate_dtype = lambda: torch.float32

    pkg.utils = utils
    pkg.model_management = model_management
    return pkg, utils, model_management


def _build_comfy_extras():
    pkg = types.ModuleType("comfy_extras")
    pkg.__path__ = []
    nodes_mask = types.ModuleType("comfy_extras.nodes_mask")

    class ImageCompositeMasked:
        @classmethod
        def execute(cls, destination, source, x, y, resize_source, mask=None, **kwargs):
            out = destination.clone()
            height, width = source.shape[1], source.shape[2]
            out[:, y:y + height, x:x + width, :] = source
            return (out,)

    nodes_mask.ImageCompositeMasked = ImageCompositeMasked
    pkg.nodes_mask = nodes_mask
    return pkg, nodes_mask


if _missing("folder_paths"):
    _install("folder_paths", _build_folder_paths())

if _missing("nodes"):
    _install("nodes", _build_nodes())

if _missing("comfy.utils"):
    _comfy, _comfy_utils, _comfy_mm = _build_comfy_package()
    _install("comfy", _comfy)
    _install("comfy.utils", _comfy_utils)
    _install("comfy.model_management", _comfy_mm)

if _missing("comfy_extras.nodes_mask"):
    _extras, _extras_mask = _build_comfy_extras()
    _install("comfy_extras", _extras)
    _install("comfy_extras.nodes_mask", _extras_mask)

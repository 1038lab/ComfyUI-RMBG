from pathlib import Path
import sys
import os
import importlib.util

__version__ = "3.0.0"

# Locate current and node directories
current_dir = Path(__file__).parent
nodes_dir = current_dir / "py"

# Add both current and nodes directories to sys.path
for path in [current_dir, nodes_dir]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Initialize mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

def load_nodes():
    """Automatically discover and load node definitions"""
    for base in (current_dir, nodes_dir):
        if not base.exists():
            continue
        for file in base.rglob("*.py"):
            if file.stem == "__init__":
                continue
            models_dir = current_dir / "models"
            try:
                if file.is_relative_to(models_dir):
                    continue
            except Exception:
                if str(models_dir) in str(file):
                    continue

            try:
                spec = importlib.util.spec_from_file_location(file.stem, file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, "NODE_CLASS_MAPPINGS"):
                        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

                    if hasattr(module, "Paths") and hasattr(module.Paths, "LLM_DIR"):
                        os.makedirs(module.Paths.LLM_DIR, exist_ok=True)
            except Exception as e:
                print(f"Error loading {file}: {e}")

# Load all nodes
load_nodes()

__all__ = ["NODE_CLASS_MAPPINGS","NODE_DISPLAY_NAME_MAPPINGS"]

print(f'\033[34m[ComfyUI-RMBG]\033[0m v\033[93m{__version__}\033[0m | '
      f'\033[93m{len(NODE_CLASS_MAPPINGS)} nodes\033[0m \033[92mLoaded\033[0m')

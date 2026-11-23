from pathlib import Path
import sys
import os
import importlib.util

__repo_name__ = "ComfyUI-RMBG"
__version__ = "3.4.0"

current_dir = Path(__file__).parent
nodes_dir = current_dir / "py"

for path in (current_dir, nodes_dir):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"


def load_nodes():
    if not nodes_dir.exists():
        return
    for file in sorted(nodes_dir.glob("*.py")):
        if file.stem == "__init__":
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
            print(f"[{__repo_name__}] Error loading {file.stem}: {e}")


load_nodes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print(f'\033[36m[{__repo_name__}]\033[0m v'
      f'\033[93m{__version__}\033[0m | '
      f'\033[37m{len(NODE_CLASS_MAPPINGS)} nodes\033[0m '
      f'\033[92mLoaded\033[0m')
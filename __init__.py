from pathlib import Path
import sys
import os
import importlib.util

__version__ = "2.9.4"

# Add module directory to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.modules[__name__] = sys.modules.get(__name__, type(__name__, (), {}))
    sys.path.insert(0, str(current_dir))

# Initialize mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

def load_nodes():
    """Automatically discover and load node definitions"""
    for file in current_dir.glob("*.py"):
        if file.stem == "__init__":
            continue
            
        try:
            # Import module
            spec = importlib.util.spec_from_file_location(file.stem, file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Update mappings
                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
                    
                # Initialize paths if available
                if hasattr(module, "Paths") and hasattr(module.Paths, "LLM_DIR"):
                    os.makedirs(module.Paths.LLM_DIR, exist_ok=True)
                    
        except Exception as e:
            print(f"Error loading {file.name}: {str(e)}")

# Load all nodes
load_nodes()

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]
# print(f'\033[34m[ComfyUI-RMBG] v{__version__} \033[92mLoaded\033[0m') 

print(f'\033[34m[ComfyUI-RMBG]\033[0m v\033[93m{__version__}\033[0m | \033[93m{len(NODE_CLASS_MAPPINGS)} nodes\033[0m \033[92mLoaded\033[0m') 

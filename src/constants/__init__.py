
from pathlib import Path

def get_project_root():
    path = Path(__file__).resolve()
    for parent in path.parents:
        # Detect project root by checking for setup.py
        if (parent / "setup.py").exists():
            return parent
    return Path.cwd()  # fallback

ROOT_DIR = get_project_root()

CONFIG_YAML_FILE = ROOT_DIR / "yamlfiles" / "config.yaml"
PARAM_YAML_FILE  = ROOT_DIR / "yamlfiles" / "param.yaml"
SECRET_YAML_FILE = ROOT_DIR / "yamlfiles" / "secrets.yaml"

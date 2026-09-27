from pathlib import Path
import os


if os.path.exists("/app/models"):
    PROJECT_ROOT = Path("/app")
else:
    # Tu lógica local de desarrollo
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data/"
NOTEBOOKS_DIR= PROJECT_ROOT / "notebooks/"
CONFIG_PATH= PROJECT_ROOT / "config.yaml"
MODEL_DIR= PROJECT_ROOT / "models/"

MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0")
MODEL_PATH = MODEL_DIR / f"model_{MODEL_VERSION}"

MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
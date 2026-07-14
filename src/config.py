from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Dataset path (not included in this repository)
DATASET_PATH = DATA_DIR / "pyg_molecular_dataset_enriched_with_xyz.pt"

# Random seed
SEED = 42

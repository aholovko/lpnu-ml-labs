"""
Paths for data, models, reports, and figures.
"""

from pathlib import Path

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent.absolute()

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

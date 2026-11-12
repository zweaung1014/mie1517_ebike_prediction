"""Paths for the model-search package. Nothing personal: everything comes from env vars or this folder.

  TEAM_TABLE    the data-v1 training table (default: data/processed/training_table.parquet at the root of the
                team repo, where notebooks/build_dataset.ipynb writes it)
  MIE1517_DATA  a work folder for saved weights and predictions (default: work/ inside this package; not in git)
"""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]  # this package folder
REPO_ROOT = PROJECT_DIR.parent                      # the team repo
TEAM_TABLE = Path(os.environ.get("TEAM_TABLE", REPO_ROOT / "data" / "processed" / "training_table.parquet"))
DATA_DIR = Path(os.environ.get("MIE1517_DATA", PROJECT_DIR / "work"))
FIG_DIR = PROJECT_DIR / "reports" / "figures"
TAB_DIR = PROJECT_DIR / "reports" / "tables"
LOCAL_TZ = "America/Toronto"

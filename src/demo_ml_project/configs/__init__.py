
# src/config.py  or config/__init__.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent   # project root
DATA_DIR = BASE_DIR / "data"

RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

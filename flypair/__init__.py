"""flypair: N connectome-simulated fruit flies in a shared closed-loop world."""
from pathlib import Path

__version__ = "0.1.0"
PKG_ROOT = Path(__file__).resolve().parent.parent
GROUPS_DIR = PKG_ROOT / "groups"
SCENARIOS_DIR = PKG_ROOT / "scenarios"

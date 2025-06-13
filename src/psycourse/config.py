from pathlib import Path

SRC = Path(__file__).parent.resolve()  # src/psycourse
DATA_DIR = SRC / "data"

ROOT = SRC.parent.parent.resolve()
BLD = ROOT / "bld"
BLD_DATA = BLD / "data"
BLD_MODELS = BLD / "models"
BLD_RESULTS = BLD / "results"

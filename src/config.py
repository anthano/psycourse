"""All the general configuration of the project."""

from pathlib import Path

SRC = Path(__file__).parent.resolve()
ROOT = SRC.joinpath("..", "..").resolve()

BLD = ROOT.joinpath("bld").resolve()


TEMPLATE_GROUPS = ["marital_status", "highest_qualification"]

# need to discuss path to data (or cleaned data) once you have R scripts etc.
# DATA_PATH_SERVER = ROOT.parent.parent.parent / "data" / "psycourse"

from pathlib import Path

SRC = Path(__file__).parent.resolve()  # src/psycourse
DATA_DIR = SRC / "data"

ROOT = SRC.parent.parent.resolve()
BLD = ROOT / "bld"
BLD_DATA = BLD / "data"
BLD_MODELS = BLD / "models"
BLD_RESULTS = BLD / "results"

WRITING = Path("/Users/anojat/Documents/psycourse-writing")

###############################################################################
# Plot design — edit here to change visuals project-wide
###############################################################################

# --- Significance tier colors ------------------------------------------------

# PRS figures
PLOT_PRS_FDR = "#6e2490"  # FDR < 0.05
PLOT_PRS_NOM = "#b87ac8"  # p < 0.05, FDR ≥ 0.05
PLOT_PRS_NS = "#f4edf6"  # p ≥ 0.05

# Lipid figures
PLOT_LIP_FDR = "#0e7a56"  # FDR < 0.05
PLOT_LIP_NOM = "#52c49a"  # p < 0.05, FDR ≥ 0.05
PLOT_LIP_NS = "#e8f8f2"  # p ≥ 0.05


# Combined figures (PRS + lipid together)
PLOT_COMBINED_PRS = "#b87ac8"
PLOT_COMBINED_LIP = "#52c49a"
PLOT_COMBINED_SHARED = "#f4a261"

# --- Geometry ----------------------------------------------------------------

PLOT_FOREST_POINT_SIZE = 90  # scatter marker area in forest (CI) plots  (pts²)
PLOT_DOT_POINT_SIZE = 80  # scatter marker area in dot / lollipop plots (pts²)
PLOT_ELINEWIDTH = 1  # CI whisker line width
PLOT_CAPSIZE = 3  # CI cap length  (pts)
PLOT_CAPTHICK = 1  # CI cap thickness
PLOT_ECOLOR = "k"  # CI whisker color — always black
PLOT_EDGECOLOR = "k"  # marker edge color

# --- Typography --------------------------------------------------------------

PLOT_FONT_FAMILY = "DejaVu Sans"
PLOT_FONTSIZE_TITLE = 16
PLOT_FONTSIZE_LABEL = 14
PLOT_FONTSIZE_TICK = 14
PLOT_FONTSIZE_LEGEND = 14

# Convenience dict accepted by plt.rcParams.update() and sns.set_theme(rc=...)
PLOT_RCPARAMS = {
    "font.family": PLOT_FONT_FAMILY,
    "axes.titlesize": PLOT_FONTSIZE_TITLE,
    "axes.labelsize": PLOT_FONTSIZE_LABEL,
    "xtick.labelsize": PLOT_FONTSIZE_TICK,
    "ytick.labelsize": PLOT_FONTSIZE_TICK,
    "legend.fontsize": PLOT_FONTSIZE_LEGEND,
}

# --- Chrome ------------------------------------------------------------------

PLOT_GRID_COLOR = "0.9"  # x-axis grid colour in forest / dot plots
PLOT_SPINE_COLOR = "#2b2b2b"  # axis spines and tick text
PLOT_GRID_DARK = "#D9D9D9"  # grid colour in compact single-column figures

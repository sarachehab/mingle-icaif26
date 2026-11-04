import matplotlib.text as mtext
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
import seaborn as sns
import scienceplots
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerBase

class LatexSizes:
    COLUMN = 3.3
    DOUBLE_COLUMN = 7.0
    HEIGHT = 8.6


def configure_matplotlib():
    plt.style.use('science')
    plt.rcParams.update({
        # --- LaTeX-style fonts ---
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.serif": ["Computer Modern Roman"],

        # --- Font sizes (LaTeX-like proportions) ---
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.titlesize": 9,

        # --- Figure  ---
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.format": "pdf",

        # --- Grid defaults ---
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "xtick.top": False,
        "ytick.right": False,

        # --- Tick defaults ---
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,

        # Scientific notation thresholds
        "axes.formatter.limits": (-4, 4),

        # --- Legend defaults ---
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": False,

        # --- Date formatting ---
        "date.converter": "concise",
    })
    sns.set_palette("tab10")


class LegendTitle(HandlerBase):
    """Renders a string handle as a bold section header in a legend."""

    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super().__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        props = {"fontsize": fontsize, **self.text_props}
        # usetex ignores fontweight: ask LaTeX for bold directly (blank padding stays blank).
        usetex = plt.rcParams["text.usetex"]
        text = rf"\textbf{{{orig_handle}}}" if usetex and orig_handle else orig_handle
        t = mtext.Text(0, 0, text, fontweight="bold", **props)
        # Reserve the header's real width, so the legend frame and tight bbox include it.
        if text:
            width = TextPath((0, 0), text, size=props["fontsize"], usetex=usetex,
                             prop=FontProperties(weight="bold")).get_extents().width
            handlebox.width = max(handlebox.width, width)
        handlebox.add_artist(t)
        return t

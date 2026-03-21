# pineapple_pastel.py

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from cycler import cycler
from contextlib import contextmanager


STYLE_DIR = Path(__file__).resolve().parent
FONT_DIR = STYLE_DIR / "assets" / "fonts"
_INTER_REGISTERED = False


def _register_inter_fonts():
    global _INTER_REGISTERED
    if _INTER_REGISTERED:
        return True
    if not FONT_DIR.exists():
        return False

    font_files = list(FONT_DIR.glob("Inter-*.ttf"))
    if not font_files:
        return False

    for font_file in font_files:
        font_manager.fontManager.addfont(str(font_file))

    _INTER_REGISTERED = True
    return True


def _base_rcparams():
    return {
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.transparent": False,
        "figure.frameon": True,
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.8,
        "legend.frameon": True,
        "legend.fancybox": True,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }


def _theme_rcparams(theme):
    if theme == "light":
        return {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "axes.edgecolor": "#444444",
            "axes.titlecolor": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "grid.color": "#C9CDD6",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.6,
            "legend.edgecolor": "#D9DCE3",
            "legend.facecolor": "#F7F8FB",
            "axes.prop_cycle": cycler(
                "color",
                [
                    "#7E8CE0",
                    "#E59ACB",
                    "#8FC1A9",
                    "#F2B880",
                    "#C3A6E8",
                    "#8FB8DE",
                ],
            ),
        }

    if theme == "dark":
        return {
            "figure.facecolor": "#16181D",
            "axes.facecolor": "#16181D",
            "savefig.facecolor": "#16181D",
            "text.color": "#EAEAF2",
            "axes.labelcolor": "#EAEAF2",
            "axes.edgecolor": "#B8BCC8",
            "axes.titlecolor": "#F5F5F7",
            "xtick.color": "#C9CDD6",
            "ytick.color": "#C9CDD6",
            "grid.color": "#545B6B",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.5,
            "legend.edgecolor": "#545B6B",
            "legend.facecolor": "#232730",
            "axes.prop_cycle": cycler(
                "color",
                [
                    "#A8B4FF",
                    "#FFB3DE",
                    "#A9D8C2",
                    "#FFD0A1",
                    "#D6BCFF",
                    "#A8CFF2",
                ],
            ),
        }

    raise ValueError("theme must be 'light' or 'dark'")


def _text_rcparams(usetex, inter_available):
    family = "Inter" if inter_available else "DejaVu Sans"

    if usetex:
        return {
            "text.usetex": True,
            "text.latex.preamble": (
                r"\usepackage{amsmath}"
                r"\usepackage{amssymb}"
                r"\usepackage{amsfonts}"
            ),
            "font.family": "serif",
            "font.serif": ["New Century Schoolbook"]
        }

    return {
        "text.usetex": False,
        "font.family": family,
        "mathtext.fontset": "stix",
        "mathtext.default": "regular",
    }


def _colormap_rcparams(colormap="cool"):
    return {
        "image.cmap": colormap,
    }


def pineapple_rcparams(theme="light", usetex=False, colormap="cool"):
    inter_available = _register_inter_fonts()

    rc = {}
    rc.update(_base_rcparams())
    rc.update(_theme_rcparams(theme))
    rc.update(_text_rcparams(usetex, inter_available))
    rc.update(_colormap_rcparams(colormap))
    return rc


@contextmanager
def pineapple_style_context(theme="light", usetex=False, colormap="cool"):
    with mpl.rc_context(
            pineapple_rcparams(theme=theme, usetex=usetex, colormap=colormap)
    ):
        yield


def pineapple_style(theme="light", usetex=False, colormap="cool"):
    mpl.rcParams.update(
        pineapple_rcparams(theme=theme, usetex=usetex, colormap=colormap)
    )

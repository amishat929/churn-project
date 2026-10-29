"""Shared colours so the notebook plots, saved PNGs and Streamlit charts match.

Palette follows a validated categorical order: slot 1 blue (customers who stay),
slot 2 orange (customers who churn). Neutrals are for chrome only.
"""

STAY = "#2a78d6"      # categorical slot 1
CHURN = "#eb6834"     # categorical slot 2
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Single-hue sequential ramp (blue, light -> dark) for magnitude
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# One colour per model, fixed order (never re-assigned by rank)
MODEL_COLORS = {
    "Logistic Regression": "#2a78d6",
    "Random Forest": "#eb6834",
    "XGBoost": "#1baf7a",
}

RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}


def style_axes(ax):
    """Recessive grid, no top/right spines, muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)

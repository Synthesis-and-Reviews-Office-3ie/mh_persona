"""
Generate publication-quality SVG diagrams for the manuscript.
Replaces ASCII art in §3.1, §3.3, §3.4, §3.5, and §5.

Output: 04_results/figures/fig_{1..5}.svg
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "04_results", "figures")
os.makedirs(OUT, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────
FONT = "Arial"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT, "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.5,
    "figure.dpi": 300,
})

C_BG      = "#F7F9FC"
C_BOX     = "#E8EDF3"
C_ACCENT  = "#2C5F8A"
C_ACCENT2 = "#D4563A"
C_TEXT    = "#1A1A2E"
C_ARROW   = "#4A6FA5"
C_GREEN   = "#2E8B57"
C_RED     = "#C0392B"
C_ORANGE  = "#E67E22"
C_GRAY    = "#7F8C8D"


def box(ax, x, y, w, h, text, color=C_BOX, text_color=C_TEXT, fontsize=8,
        edgecolor=C_ACCENT, linewidth=0.8, ha="center", va="center", bold=False):
    """Draw a rounded box with centered text."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=edgecolor, linewidth=linewidth,
        transform=ax.transData, zorder=2,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha=ha, va=va,
            fontsize=fontsize, color=text_color, weight=weight, zorder=3)
    return rect


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.2, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=1)


# ══════════════════════════════════════════════════════════════
# Figure 1: Steering Vector Extraction (§3.1)
# ══════════════════════════════════════════════════════════════
def fig1_steering_extraction():
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(5, 3.9, "Steering Vector Extraction", ha="center", fontsize=11,
            weight="bold", color=C_ACCENT)

    # HIGH path
    box(ax, 0.2, 2.6, 2.0, 0.7, "HIGH prompt\n(empathetic)", color="#D5E8D4",
        edgecolor=C_GREEN, fontsize=7.5)
    box(ax, 3.0, 2.6, 2.0, 0.7, "Forward pass\n→ Layer L", fontsize=7.5)
    box(ax, 5.8, 2.6, 2.0, 0.7, "Activation\n$\\mathbf{h}_{\\mathrm{high}}$",
        fontsize=7.5, color="#D5E8D4", edgecolor=C_GREEN)

    arrow(ax, 2.2, 2.95, 3.0, 2.95)
    arrow(ax, 5.0, 2.95, 5.8, 2.95)

    # LOW path
    box(ax, 0.2, 0.9, 2.0, 0.7, "LOW prompt\n(detached)", color="#F8D7DA",
        edgecolor=C_RED, fontsize=7.5)
    box(ax, 3.0, 0.9, 2.0, 0.7, "Forward pass\n→ Layer L", fontsize=7.5)
    box(ax, 5.8, 0.9, 2.0, 0.7, "Activation\n$\\mathbf{h}_{\\mathrm{low}}$",
        fontsize=7.5, color="#F8D7DA", edgecolor=C_RED)

    arrow(ax, 2.2, 1.25, 3.0, 1.25)
    arrow(ax, 5.0, 1.25, 5.8, 1.25)

    # Difference arrow → steering vector
    arrow(ax, 7.3, 2.6, 7.3, 1.6, color=C_ACCENT, lw=1.5, style="<->")
    ax.text(7.5, 2.1, "Δ", fontsize=10, color=C_ACCENT, weight="bold")

    box(ax, 8.2, 1.5, 1.6, 0.9, "Steering\nVector\n$\\hat{\\mathbf{v}}$",
        color=C_ACCENT, text_color="white", fontsize=8, bold=True)

    arrow(ax, 7.8, 2.0, 8.2, 1.95, color=C_ACCENT, lw=1.5)

    # Bottom note
    ax.text(5, 0.25, "Repeat × 5 contrast pairs → Average → Normalise\n"
            "Layer selected by best behavioural correlation (not Cohen's d)",
            ha="center", fontsize=7, color=C_GRAY, style="italic")

    fig.savefig(os.path.join(OUT, "fig1_steering_extraction.svg"),
                bbox_inches="tight", pad_inches=0.15)
    fig.savefig(os.path.join(OUT, "fig1_steering_extraction.pdf"),
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  ✓ Figure 1: Steering vector extraction")


# ══════════════════════════════════════════════════════════════
# Figure 2: Diagnostic Results (§3.3)
# ══════════════════════════════════════════════════════════════
def fig2_diagnostics():
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(4, 4.1, "What Predicts Steering Success?", ha="center",
            fontsize=11, weight="bold", color=C_ACCENT)

    # Failed predictors
    ax.plot(0.4, 3.3, marker="x", markersize=10, color=C_RED, markeredgewidth=2.5)
    ax.text(0.7, 3.3, "Activation separation (Cohen's d)",
            fontsize=8, color=C_GRAY, va="center")
    ax.text(6.5, 3.3, "r = -0.374", fontsize=8, color=C_GRAY,
            va="center", family="monospace")

    ax.plot(0.4, 2.7, marker="x", markersize=10, color=C_RED, markeredgewidth=2.5)
    ax.text(0.7, 2.7, "Within-class variance",
            fontsize=8, color=C_GRAY, va="center")
    ax.text(6.5, 2.7, "r = -0.370", fontsize=8, color=C_GRAY,
            va="center", family="monospace")

    ax.plot(0.4, 2.1, marker="x", markersize=10, color=C_RED, markeredgewidth=2.5)
    ax.text(0.7, 2.1, "Prompt consistency (cosine sim)",
            fontsize=8, color=C_GRAY, va="center")
    ax.text(6.5, 2.1, "r = +0.386", fontsize=8, color=C_GRAY,
            va="center", family="monospace")

    # Divider
    ax.plot([0.3, 7.7], [1.55, 1.55], color=C_ACCENT, lw=0.5, ls="--")

    # Success predictor
    ax.plot(0.4, 0.9, marker="o", markersize=10, color=C_GREEN, markeredgewidth=2.5, fillstyle="none")
    ax.plot(0.4, 0.9, marker="$\\checkmark$", markersize=7, color=C_GREEN)
    ax.text(0.7, 0.9, "Behavioural difference (judge delta)",
            fontsize=9, color=C_TEXT, va="center", weight="bold")
    ax.text(6.5, 0.9, "r = +0.899***", fontsize=9, color=C_GREEN,
            va="center", family="monospace", weight="bold")

    ax.text(4, 0.2, "N = 24 trait × model combinations",
            ha="center", fontsize=7, color=C_GRAY, style="italic")

    fig.savefig(os.path.join(OUT, "fig2_diagnostics.svg"),
                bbox_inches="tight", pad_inches=0.15)
    fig.savefig(os.path.join(OUT, "fig2_diagnostics.pdf"),
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  ✓ Figure 2: Diagnostic results")


# ══════════════════════════════════════════════════════════════
# Figure 3: Contrastive Probing Pipeline (§3.4)
# ══════════════════════════════════════════════════════════════
def fig3_contrastive_probing():
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(5, 5.1, "Contrastive Probing Pipeline", ha="center",
            fontsize=11, weight="bold", color=C_ACCENT)

    steps = [
        ("(1)  Generate 500 responses\n       to mental health scenarios", 4.0),
        ("(2)  Score each response (GPT-4o-mini\n       judge, 1-7 scale)", 3.2),
        ("(3)  Select HIGH (>=5) and LOW (<=3)\n       scored responses", 2.4),
        ("(4)  Extract hidden states from\n       model's OWN responses", 1.6),
        ("(5)  Logistic regression weights\n       = direction vector", 0.8),
    ]

    for i, (text, y) in enumerate(steps):
        c = C_ACCENT if i == 4 else C_BOX
        ec = C_ACCENT if i == 4 else "#B0BEC5"
        tc = "white" if i == 4 else C_TEXT
        box(ax, 1.5, y, 5.0, 0.65, text, color=c, edgecolor=ec,
            text_color=tc, fontsize=7.5, ha="center")
        if i < 4:
            arrow(ax, 4.0, y, 4.0, y - 0.15, color=C_ARROW, lw=1.0)

    # Key insight box on the right
    box(ax, 7.2, 1.8, 2.5, 1.8,
        "Key insight:\nModel's own\nbehaviour defines\nthe contrastive\nclasses — not\nexternal templates",
        color="#FFF3CD", edgecolor=C_ORANGE, fontsize=7, ha="center")

    arrow(ax, 6.5, 2.4, 7.2, 2.6, color=C_ORANGE, lw=1.0, style="->")

    fig.savefig(os.path.join(OUT, "fig3_contrastive_probing.svg"),
                bbox_inches="tight", pad_inches=0.15)
    fig.savefig(os.path.join(OUT, "fig3_contrastive_probing.pdf"),
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  ✓ Figure 3: Contrastive probing pipeline")


# ══════════════════════════════════════════════════════════════
# Figure 4: Real-Time Monitoring Architecture (§3.5)
# ══════════════════════════════════════════════════════════════
def fig4_monitoring():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(5.5, 6.1, "Real-Time Monitoring Architecture", ha="center",
            fontsize=11, weight="bold", color=C_ACCENT)

    # Input
    box(ax, 0.2, 4.5, 2.3, 0.7, "User message", fontsize=8)
    arrow(ax, 2.5, 4.85, 3.2, 4.85)

    # Model
    box(ax, 3.2, 4.5, 2.5, 0.7, "Model generates\nresponse", fontsize=8,
        color="#D5E8D4", edgecolor=C_GREEN)
    arrow(ax, 5.7, 4.85, 6.3, 4.85)

    # Extract activations
    box(ax, 6.3, 4.5, 2.8, 0.7, "Extract activations\nat validated layers",
        fontsize=7.5, color=C_BOX)

    # Projection fan-out
    arrow(ax, 7.7, 4.5, 6.0, 3.7, color=C_ARROW, lw=0.8)
    arrow(ax, 7.7, 4.5, 7.7, 3.7, color=C_ARROW, lw=0.8)
    arrow(ax, 7.7, 4.5, 9.4, 3.7, color=C_ARROW, lw=0.8)

    # Trait projections
    traits = [
        ("empathetic_resp.", 4.7, 3.0),
        ("crisis_recog.", 6.4, 3.0),
        ("... (8 traits)", 8.1, 3.0),
    ]
    for label, x, y in traits:
        box(ax, x, y, 1.5, 0.6, f"Project →\n{label}",
            fontsize=6.5, color="#E8E0F0", edgecolor="#7B68AE")

    # Converge to SPC
    for x in [5.45, 7.15, 8.85]:
        arrow(ax, x, 3.0, 7.0, 2.3, color=C_ARROW, lw=0.8)

    box(ax, 5.5, 1.6, 3.0, 0.7, "EWMA + CUSUM\nstatistical process control",
        fontsize=7.5, color=C_BOX, edgecolor=C_ACCENT)

    # Alert levels
    alert_data = [
        ("NORMAL", C_GREEN, 0.2, "continue"),
        ("WATCH", C_ORANGE, 2.5, "|z|≥1.5 → log"),
        ("WARNING", "#E74C3C", 4.8, "|z|≥2.5 → flag"),
        ("CRITICAL", "#8B0000", 7.3, "|z|≥3.5 → intervene"),
    ]

    arrow(ax, 7.0, 1.6, 5.0, 1.0, color=C_ARROW, lw=0.8)

    for label, color, x, desc in alert_data:
        box(ax, x, 0.15, 2.1, 0.6, f"{label}\n{desc}",
            fontsize=6.5, color=color, text_color="white", edgecolor=color,
            bold=True)

    # Cost note
    ax.text(5.5, -0.35, "Cost: ONE dot product per trait per turn (≈0.001ms)",
            ha="center", fontsize=7, color=C_GRAY, style="italic")

    fig.savefig(os.path.join(OUT, "fig4_monitoring_architecture.svg"),
                bbox_inches="tight", pad_inches=0.15)
    fig.savefig(os.path.join(OUT, "fig4_monitoring_architecture.pdf"),
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  ✓ Figure 4: Real-time monitoring architecture")


# ══════════════════════════════════════════════════════════════
# Figure 5: Mistral Safety Gradient (§5)
# ══════════════════════════════════════════════════════════════
def fig5_safety_gradient():
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    layers   = [10,   12,   14,   15,   16,   18,   19]
    r_values = [-0.329, 0.679, 0.651, 0.635, 0.464, 0.137, 0.124]
    labels   = ["PROTECTIVE", "VULNERABLE", "VULNERABLE", "VULNERABLE",
                "CONCERNING", "NEUTRAL", "NEUTRAL"]
    colors   = [C_GREEN, C_RED, C_RED, C_RED, C_ORANGE, C_GRAY, C_GRAY]
    harms    = ["", "harm=8.0", "harm=6.4", "harm=4.0", "", "", ""]

    bars = ax.barh(range(len(layers)), r_values, color=colors, edgecolor="white",
                   height=0.65, zorder=2)

    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {l}" for l in layers], fontsize=8)
    ax.set_xlabel("Correlation with harmfulness (r)", fontsize=9)
    ax.set_title("Mistral-7B: Layer-Resolved Safety Profile",
                 fontsize=10, weight="bold", color=C_ACCENT, pad=10)

    ax.axvline(0, color=C_TEXT, lw=0.5, zorder=1)
    ax.set_xlim(-0.5, 0.85)

    # Annotations
    for i, (lbl, h) in enumerate(zip(labels, harms)):
        x_pos = r_values[i]
        offset = 0.03 if x_pos >= 0 else -0.03
        ha = "left" if x_pos >= 0 else "right"
        txt = lbl
        if h:
            txt += f"  ({h})"
        ax.text(x_pos + offset, i, txt, va="center", ha=ha,
                fontsize=6.5, color=colors[i], weight="bold")

    # Note
    ax.text(0.5, -0.22,
            "At coeff +5.0 (beyond validated ±3.0 range). Qwen2 resisted all manipulation.",
            transform=ax.transAxes, ha="center", fontsize=7,
            color=C_GRAY, style="italic")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    fig.savefig(os.path.join(OUT, "fig5_safety_gradient.svg"),
                bbox_inches="tight", pad_inches=0.15)
    fig.savefig(os.path.join(OUT, "fig5_safety_gradient.pdf"),
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  ✓ Figure 5: Mistral safety gradient")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating publication figures...")
    fig1_steering_extraction()
    fig2_diagnostics()
    fig3_contrastive_probing()
    fig4_monitoring()
    fig5_safety_gradient()
    print(f"\nAll figures saved to: {os.path.abspath(OUT)}")
    print("Formats: SVG (vector) + PDF (print)")

"""
Kinexica Website — Diagram Generator
Produces all 4 data-driven diagrams as PNG files for the website.
Run: python website/generate_diagrams.py
"""
# pylint: disable=invalid-name
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

import matplotlib
matplotlib.use("Agg")  # must be called before importing pyplot

OUT = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "public", "diagrams")
os.makedirs(OUT, exist_ok=True)

BG = "#07070a"
CARD = "#111116"
BORDER = "#1e1e28"
GREEN = "#10b981"
BLUE = "#3b82f6"
PURPLE = "#8b5cf6"
RED = "#ef4444"
AMBER = "#f59e0b"
WHITE = "#f4f4f5"
GREY = "#71717a"
FONT = "DejaVu Sans"

plt.rcParams.update({
    "font.family":     FONT,
    "text.color":      WHITE,
    "axes.labelcolor": WHITE,
    "xtick.color":     GREY,
    "ytick.color":     GREY,
    "figure.facecolor": BG,
    "axes.facecolor":   CARD,
    "axes.edgecolor":   BORDER,
    "grid.color":       BORDER,
    "grid.linewidth":   0.6,
})


# ════════════════════════════════════════════════════════════════════════════
# DIAGRAM 1 — 10-Year Economic Analysis (Bar Chart)
# ════════════════════════════════════════════════════════════════════════════
def diagram_economic():
    """10-year cumulative financial impact: Traditional vs Kinexica."""
    years = np.arange(0, 11)
    traditional = [-10, -30, -55, -70, -90, -115, -140, -155, -175, -195, -220]
    kinexica = [-80, -60, -35, 0, 45, 125, 200, 265, 340, 430, 535]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)

    w = 0.38
    bars_trad = ax.bar(years - w/2, traditional, w, color=BLUE,
                       alpha=0.82, zorder=3, label="Traditional Model")
    bars_kinx = ax.bar(years + w/2, kinexica,    w, color=AMBER,
                       alpha=0.90, zorder=3, label="Kinexica Model")

    # Value labels on top/bottom of each bar
    for trad_rect in bars_trad:
        h = trad_rect.get_height()
        ax.text(
            trad_rect.get_x() + trad_rect.get_width() / 2,
            h - (8 if h < 0 else 8),
            f"${int(h)}k",
            ha="center", va="top" if h < 0 else "bottom",
            fontsize=7.5, color=BLUE, alpha=0.9,
        )
    for kinx_rect in bars_kinx:
        h = kinx_rect.get_height()
        ax.text(
            kinx_rect.get_x() + kinx_rect.get_width() / 2,
            h + (5 if h >= 0 else -5),
            f"${int(h)}k",
            ha="center", va="bottom" if h >= 0 else "top",
            fontsize=7.5, color=AMBER, alpha=0.9,
        )

    ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xlabel("Year",  fontsize=13, labelpad=10, color=WHITE)
    ax.set_ylabel("Cumulative Financial Impact ($K)",
                  fontsize=12, labelpad=10, color=WHITE)
    ax.set_title("10-Year Economic Analysis — Traditional vs Kinexica", fontsize=16,
                 fontweight="bold", color=WHITE, pad=18)
    ax.set_xticks(years)
    ax.set_xticklabels([f"Yr {y}" for y in years], fontsize=10)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${int(x)}k"))
    ax.grid(axis="y", zorder=1)
    ax.legend(fontsize=11, loc="upper left", facecolor=CARD, edgecolor=BORDER,
              labelcolor=WHITE, framealpha=0.9)

    # Annotation: break-even
    ax.annotate("Break-even\n≈ Year 3", xy=(3, 0), xytext=(3.6, 70),
                fontsize=9.5, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4),
                bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=GREEN, alpha=0.8))

    fig.tight_layout(pad=1.5)
    path = os.path.join(OUT, "economic_analysis.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✔  {path}")


# ════════════════════════════════════════════════════════════════════════════
# DIAGRAM 2 — Why We Are Best (Radar / Spider Chart)
# ════════════════════════════════════════════════════════════════════════════
def diagram_radar():
    """Capability radar: Kinexica vs Traditional vs IoT Sensors only."""
    categories = [
        "Shelf Life\nPrediction", "Pathogen\nDetection", "Fraud\nDetection",
        "Blockchain\nSettlement", "Real-time\nAlerts", "Carbon\nCredits"
    ]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    kinexica = [97, 92, 95, 98, 94, 88]
    traditional = [30, 20, 10,  5, 25, 15]
    iot_only = [65, 40, 20, 12, 70, 10]

    kinexica += kinexica[:1]
    traditional += traditional[:1]
    iot_only += iot_only[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)

    ax.plot(angles, kinexica,    color=GREEN,
            linewidth=2.2, linestyle="solid")
    ax.fill(angles, kinexica,    alpha=0.25,   color=GREEN)
    ax.plot(angles, traditional, color=BLUE,
            linewidth=1.6, linestyle="dashed")
    ax.fill(angles, traditional, alpha=0.12,   color=BLUE)
    ax.plot(angles, iot_only,    color=PURPLE,
            linewidth=1.6, linestyle="dotted")
    ax.fill(angles, iot_only,    alpha=0.12,   color=PURPLE)

    ax.set_thetagrids(np.degrees(angles[:-1]),
                      categories, fontsize=11, color=WHITE)
    ax.set_ylim(0, 100)
    ax.set_yticklabels([])
    ax.set_rlabel_position(30)
    for r in [20, 40, 60, 80, 100]:
        ax.plot(angles, [r]*len(angles), color=BORDER, linewidth=0.5, zorder=0)

    ax.set_title("Capability Benchmark — Kinexica vs Competitors",
                 fontsize=14, fontweight="bold", color=WHITE, pad=28)

    patches = [
        mpatches.Patch(
            color=GREEN,  label="Kinexica (PINN + CV + Blockchain)"),
        mpatches.Patch(color=BLUE,   label="Traditional Cold-Chain"),
        mpatches.Patch(color=PURPLE, label="IoT Sensors Only"),
    ]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.14),
              fontsize=10, facecolor=CARD, edgecolor=BORDER, labelcolor=WHITE,
              framealpha=0.9, ncol=1)

    fig.tight_layout()
    path = os.path.join(OUT, "capability_radar.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✔  {path}")


# ════════════════════════════════════════════════════════════════════════════
# DIAGRAM 3 — Arrhenius Shelf-Life Decay Curves
# ════════════════════════════════════════════════════════════════════════════
def diagram_arrhenius():
    """Physics-driven shelf-life decay at 4 temperature profiles."""
    t = np.linspace(0, 200, 500)  # hours

    def shelf(t_arr, T_c, base=200.0):
        T_k = T_c + 273.15
        k = 1e8 * np.exp(-50000 / (8.314 * T_k))
        return np.maximum(base * np.exp(-k * t_arr * 0.01), 0)

    configs = [
        (5,  GREEN,  "5 °C  — Optimal Cold"),
        (15, BLUE,   "15 °C — Room (Stable)"),
        (25, AMBER,  "25 °C — Warm (Risk)"),
        (35, RED,    "35 °C — Hot (Critical)"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)

    for temp, color, label in configs:
        decay = shelf(t, temp)
        ax.plot(t, decay, color=color, linewidth=2.2, label=label)

    ax.axhline(120, color=GREY, linewidth=1, linestyle="--", alpha=0.7)
    ax.text(202, 122, "Distress\nThreshold (120h)",
            color=GREY, fontsize=9, va="bottom")

    ax.fill_between(t, 0, 120, alpha=0.04, color=RED)
    ax.fill_between(t, 120, 210, alpha=0.04, color=GREEN)

    ax.set_xlabel("Time (hours)", fontsize=13, color=WHITE)
    ax.set_ylabel("Estimated Shelf Life (hours)", fontsize=12, color=WHITE)
    ax.set_title("Arrhenius-Driven Decay — PINN Physics Constraint",
                 fontsize=15, fontweight="bold", color=WHITE, pad=14)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 210)
    ax.grid(True, zorder=1)
    ax.legend(fontsize=10.5, facecolor=CARD, edgecolor=BORDER,
              labelcolor=WHITE, framealpha=0.9)

    fig.tight_layout(pad=1.5)
    path = os.path.join(OUT, "arrhenius_decay.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✔  {path}")


# ════════════════════════════════════════════════════════════════════════════
# DIAGRAM 4 — Market Opportunity (Donut Chart)
# ════════════════════════════════════════════════════════════════════════════
def diagram_market():
    """TAM breakdown by sector."""
    labels = ["B2B\nAgri-Supply", "B2C\nConsumers", "B2G\nGovernment",
              "Aquaculture", "Carbon Credits"]
    sizes = [40, 22, 18, 12, 8]
    colors = [GREEN, BLUE, PURPLE, AMBER, "#06b6d4"]
    explode = [0.04, 0.02, 0.02, 0.02, 0.02]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    _wedges, _texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.0f%%", startangle=140,
        pctdistance=0.72, labeldistance=1.12,
        wedgeprops=dict(linewidth=1.8, edgecolor=BG),
        textprops=dict(color=WHITE, fontsize=11),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color(BG)

    # Draw hole for donut
    centre_circle = plt.Circle((0, 0), 0.52, fc=BG)
    ax.add_artist(centre_circle)
    ax.text(0, 0.06, "$2.4T", ha="center", va="center",
            fontsize=20, fontweight="bold", color=GREEN)
    ax.text(0, -0.18, "Global TAM", ha="center", va="center",
            fontsize=11, color=GREY)

    ax.set_title("Market Opportunity Breakdown — Total Addressable Market",
                 fontsize=14, fontweight="bold", color=WHITE, pad=18)
    fig.tight_layout()
    path = os.path.join(OUT, "market_opportunity.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✔  {path}")


# ════════════════════════════════════════════════════════════════════════════
# DIAGRAM 5 — PINN Architecture (Text-based flow diagram)
# ════════════════════════════════════════════════════════════════════════════
def diagram_architecture():
    """Flowchart of the PINN system architecture."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    boxes = [
        (0.06, 0.5, BLUE,   "SENSORS\n────────\nTemp · Humidity\nEthylene · CV Vision"),
        (0.27, 0.5, PURPLE,
         "PINN ENGINE\n────────────\nArrhenius Physics\nNeural Inference (5→128→64→2)"),
        (0.50, 0.5, GREEN,  "PREDICTIONS\n────────────\nShelf Life (hrs)\nPIDR · Status"),
        (0.72, 0.5, AMBER,  "AI BROKER\n────────────\nSwarm Negotiation\nAuto-Liquidation"),
        (0.92, 0.5, "#06b6d4", "BLOCKCHAIN\n────────────\nWeb3 Settlement\nSynthi-ID Stamp"),
    ]

    for (x, y, color, text) in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.085, y - 0.32), 0.17, 0.64,
            boxstyle="round,pad=0.015", linewidth=1.8,
            edgecolor=color, facecolor=CARD, zorder=2
        ))
        ax.text(x, y + 0.02, text, ha="center", va="center",
                fontsize=9.2, color=WHITE, zorder=3,
                linespacing=1.55)
        # Glow dot on top
        ax.plot(x, y + 0.35, "o", color=color, markersize=9, zorder=4)

    # Arrows between boxes
    arrow_x = [0.145, 0.355, 0.585, 0.805]
    for ax_x in arrow_x:
        ax.annotate("", xy=(ax_x + 0.01, 0.5), xytext=(ax_x - 0.01, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=GREY,
                                    lw=1.6, mutation_scale=18))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Kinexica System Architecture — End-to-End Pipeline",
                 fontsize=15, fontweight="bold", color=WHITE, pad=14)
    footer_txt = (
        "Physics-Informed Neural Network (PINN)  ·  Computer Vision"
        "  ·  AI Agent Swarm  ·  Web3 Smart Contracts"
    )
    ax.text(0.5, 0.06, footer_txt, ha="center", fontsize=9.5, color=GREY)

    fig.tight_layout(pad=1.2)
    path = os.path.join(OUT, "architecture.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✔  {path}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nGenerating Kinexica website diagrams...\n")
    diagram_economic()
    diagram_radar()
    diagram_arrhenius()
    diagram_market()
    diagram_architecture()
    print("\nAll diagrams saved to:", OUT)

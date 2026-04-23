from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
MANUSCRIPT_FIGURES = ROOT.parent / "RockJointNet_TNNLS_manuscript" / "figures"


def arrow(ax, a, b, lw=1.4, style="-|>", dashed=False, rad=0.0):
    patch = FancyArrowPatch(
        a,
        b,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color="0.15",
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(patch)
    return patch


def node(ax, xy, text, r=0.038, fs=14, lw=1.3):
    circ = Circle(xy, r, facecolor="white", edgecolor="0.1", linewidth=lw)
    ax.add_patch(circ)
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs)
    return circ


def box(ax, xy, w, h, text, fs=13, dashed=False, lw=1.2):
    rect = Rectangle(
        xy,
        w,
        h,
        facecolor="white",
        edgecolor="0.1",
        linewidth=lw,
        linestyle=(0, (2, 2)) if dashed else "solid",
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)
    return rect


def draw() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.6, 7.0), dpi=240)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Titles.
    ax.text(0.06, 0.94, r"$\bar{\sigma}_n,\bar{\mathbf{c}}$", fontsize=18, ha="center")
    ax.text(0.275, 0.895, "NONDIMENSIONALISATION", fontsize=12, ha="center")

    # Raw-to-nondimensional flow on the left.
    ax.text(0.06, 0.82, r"$\sigma_n,\mathbf{c}$", fontsize=18, ha="center")
    arrow(ax, (0.06, 0.91), (0.06, 0.84), lw=1.2)
    arrow(ax, (0.06, 0.79), (0.06, 0.63), lw=1.2)
    arrow(ax, (0.10, 0.83), (0.47, 0.83), lw=1.0)
    arrow(ax, (0.47, 0.83), (0.47, 0.63), lw=1.0)

    c_node = node(ax, (0.06, 0.58), r"$c$", r=0.037)
    s_node = node(ax, (0.06, 0.45), r"$s$", r=0.037)

    # Hidden encoder/path nodes.
    enc_nodes = [
        (0.25, 0.68, r"$e$"),
        (0.25, 0.57, r"$z_c$"),
        (0.25, 0.46, r"$u_k$"),
        (0.25, 0.35, r"$q_k$"),
    ]
    for x, y, t in enc_nodes:
        node(ax, (x, y), t, r=0.040)

    # Output-side nodes.
    q_node = node(ax, (0.44, 0.57), r"$a_k$", r=0.040)
    r_node = node(ax, (0.44, 0.43), r"$r_k$", r=0.040)

    # Connections from c/s to hidden nodes.
    for _, y, _ in enc_nodes:
        arrow(ax, (0.095, 0.58), (0.21, y), lw=1.0)
        arrow(ax, (0.095, 0.45), (0.21, y), lw=1.0)

    # Dense connections among hidden nodes and rate nodes.
    for _, y, _ in enc_nodes:
        arrow(ax, (0.29, y), (0.40, 0.57), lw=1.0)
        arrow(ax, (0.29, y), (0.40, 0.43), lw=1.0)

    # Integral output.
    tau_box = box(ax, (0.56, 0.515), 0.095, 0.105, r"$\hat{\tau}_p$", fs=18)
    arrow(ax, (0.48, 0.57), (0.56, 0.57), lw=1.2)
    arrow(ax, (0.48, 0.43), (0.56, 0.54), lw=1.2)
    ax.text(0.505, 0.495, r"$\sum \omega_k r_k$", fontsize=12, ha="center")

    curve_box = box(ax, (0.56, 0.33), 0.095, 0.105, r"$\hat{y}(x)$", fs=18)
    arrow(ax, (0.48, 0.43), (0.56, 0.385), lw=1.2)
    ax.text(0.61, 0.31, "curve", fontsize=10, ha="center")

    # Right AutoDiff/physics block.
    box(ax, (0.78, 0.15), 0.16, 0.72, "", dashed=True, lw=1.0)
    ax.text(0.86, 0.83, "AutoDiff", fontsize=12, ha="center")
    phys_nodes = [
        (0.86, 0.75, r"$\partial_{\sigma}$"),
        (0.86, 0.62, r"$\tau_0$"),
        (0.86, 0.49, r"$+$"),
        (0.86, 0.36, r"$w_{BB}$"),
        (0.86, 0.23, r"$Q_{phys}$"),
    ]
    for x, y, t in phys_nodes:
        node(ax, (x, y), t, r=0.040)
    ax.text(0.86, 0.10, "Physics\nchecks", fontsize=12, ha="center")

    arrow(ax, (0.655, 0.57), (0.82, 0.75), lw=1.0)
    arrow(ax, (0.655, 0.57), (0.82, 0.62), lw=1.0)
    arrow(ax, (0.655, 0.57), (0.82, 0.49), lw=1.0)
    arrow(ax, (0.29, 0.35), (0.82, 0.36), lw=1.0)

    # Loss block at the bottom.
    box(ax, (0.56, 0.12), 0.12, 0.16, "DATA", dashed=True, fs=13)
    ax.text(0.18, 0.17, "LOSS  =", fontsize=13, ha="center")
    ax.text(0.42, 0.17, r"$L_{data}$", fontsize=15, ha="center")
    ax.text(0.52, 0.17, "+", fontsize=15, ha="center")
    ax.text(0.74, 0.17, r"$L_{BB}$", fontsize=15, ha="center")
    ax.text(0.18, 0.13, r"$L_{curve}$", fontsize=15, ha="center")

    # Loss arrows.
    arrow(ax, (0.61, 0.515), (0.61, 0.28), lw=1.0)
    arrow(ax, (0.61, 0.33), (0.61, 0.28), lw=1.0)
    arrow(ax, (0.78, 0.36), (0.72, 0.20), lw=1.0)
    arrow(ax, (0.06, 0.45), (0.06, 0.12), lw=1.0)
    arrow(ax, (0.06, 0.12), (0.34, 0.12), lw=1.0)
    arrow(ax, (0.10, 0.58), (0.21, 0.68), lw=1.0)

    # Inverse label near outputs.
    box(ax, (0.55, 0.38), 0.13, 0.22, "Inverse", dashed=True, fs=12)
    # Redraw output labels above the inverse dashed box.
    ax.text(0.6075, 0.5675, r"$\hat{\tau}_p$", ha="center", va="center", fontsize=18)
    ax.text(0.6075, 0.3825, r"$\hat{y}(x)$", ha="center", va="center", fontsize=18)

    # Parameter update line.
    arrow(ax, (0.35, 0.12), (0.06, 0.12), lw=1.0, style="<|-")
    arrow(ax, (0.06, 0.12), (0.06, 0.42), lw=1.0)

    out_png = OUT / "geospin_framework_pinn_style.png"
    out_pdf = OUT / "geospin_framework_pinn_style.pdf"
    ms_png = MANUSCRIPT_FIGURES / "geospin_framework_pinn_style.png"
    ms_pdf = MANUSCRIPT_FIGURES / "geospin_framework_pinn_style.pdf"
    for path in (out_png, out_pdf, ms_png, ms_pdf):
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    draw()


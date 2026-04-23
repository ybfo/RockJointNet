from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
MANUSCRIPT_FIGURES = ROOT.parent / "RockJointNet_TNNLS_manuscript" / "figures"


def arrow(ax, start, end, dashed=False, rad=0.0, lw=1.55):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color="0.08",
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(patch)
    return patch


def block(ax, x, y, w, h, text, fs=13, dashed=False, bold=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        facecolor="white",
        edgecolor="0.08",
        linewidth=1.3,
        linestyle=(0, (3, 2.5)) if dashed else "solid",
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        fontsize=fs,
        ha="center",
        va="center",
        weight="bold" if bold else "normal",
    )
    return patch


def draw() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.0, 5.2), dpi=280)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Headers.
    ax.text(0.06, 0.93, "Input", fontsize=13, ha="center")
    ax.text(0.22, 0.93, "Path", fontsize=13, ha="center")
    ax.text(0.43, 0.93, "Latent dynamics", fontsize=13, ha="center")
    ax.text(0.62, 0.93, "Rate", fontsize=13, ha="center")
    ax.text(0.82, 0.93, "Output", fontsize=13, ha="center")

    # Main context path.
    y1 = 0.76
    block(ax, 0.030, y1, 0.075, 0.070, r"$\mathbf{c}$", fs=17, bold=True)
    block(ax, 0.145, y1, 0.090, 0.070, r"$E_\phi$", fs=16)
    block(ax, 0.275, y1, 0.105, 0.070, r"$\mathbf{h}(0)$", fs=15)
    block(ax, 0.415, y1, 0.110, 0.070, r"$F_\theta$", fs=16)
    block(ax, 0.560, y1, 0.120, 0.070, r"$\mathbf{h}(u_k)$", fs=14)
    for a, b in [((0.105, y1 + 0.035), (0.145, y1 + 0.035)), ((0.235, y1 + 0.035), (0.275, y1 + 0.035)), ((0.380, y1 + 0.035), (0.415, y1 + 0.035)), ((0.525, y1 + 0.035), (0.560, y1 + 0.035))]:
        arrow(ax, a, b)

    # Main stress path.
    y2 = 0.56
    block(ax, 0.030, y2, 0.075, 0.070, r"$\sigma$", fs=17, bold=True)
    block(ax, 0.145, y2, 0.090, 0.070, r"$s$", fs=17)
    block(ax, 0.275, y2, 0.105, 0.070, r"$u_k$", fs=17)
    block(ax, 0.415, y2, 0.110, 0.070, r"$\sigma(u_k)$", fs=14)
    block(ax, 0.560, y2, 0.120, 0.070, r"$\mathbf{q}(u_k,\mathbf{c})$", fs=12)
    for a, b in [((0.105, y2 + 0.035), (0.145, y2 + 0.035)), ((0.235, y2 + 0.035), (0.275, y2 + 0.035)), ((0.380, y2 + 0.035), (0.415, y2 + 0.035)), ((0.525, y2 + 0.035), (0.560, y2 + 0.035))]:
        arrow(ax, a, b)

    # Formula labels, placed above arrows not across nodes.
    ax.text(0.190, 0.665, r"$s=\log(1+\sigma/\sigma_{\rm ref})$", fontsize=10.2, ha="center")
    ax.text(0.328, 0.665, r"$u_k=\frac{s}{2}(\xi_k+1)$", fontsize=10.2, ha="center")
    ax.text(0.470, 0.665, r"$\sigma_{\rm ref}(e^{u_k}-1)$", fontsize=10.2, ha="center")
    ax.text(0.544, 0.635, r"$\mathcal{Q}$", fontsize=11, ha="center")
    ax.text(0.470, 0.845, r"$d\mathbf{h}/ds=F_\theta(\mathbf{h},s,\mathbf{c},\mathbf{q})$", fontsize=10.2, ha="center")

    # Short condition arrows only.
    arrow(ax, (0.328, y2 + 0.070), (0.438, y1), dashed=True, rad=-0.05, lw=1.1)
    arrow(ax, (0.620, y2 + 0.070), (0.498, y1), dashed=True, rad=0.05, lw=1.1)

    # Rate head and integral.
    block(ax, 0.720, y1, 0.100, 0.070, r"$G_\psi$", fs=16)
    block(ax, 0.720, y2, 0.100, 0.070, r"$r_\psi$", fs=17)
    block(ax, 0.870, y2, 0.090, 0.070, r"$\widehat{\tau}_p$", fs=17, dashed=True)
    arrow(ax, (0.680, y1 + 0.035), (0.720, y1 + 0.035))
    arrow(ax, (0.680, y2 + 0.035), (0.720, y2 + 0.035))
    arrow(ax, (0.770, y1), (0.770, y2 + 0.070))
    arrow(ax, (0.820, y2 + 0.035), (0.870, y2 + 0.035))
    ax.text(0.795, 0.675, r"$\mathrm{softplus}+\varepsilon$", fontsize=10.2, ha="center")
    ax.text(0.845, 0.505, r"$\frac{s}{2}\sum_k\omega_k r_\psi(u_k,\mathbf{c})$", fontsize=10.4, ha="center")

    # Guarantees: one small right-side box, one arrow from output.
    block(ax, 0.800, 0.735, 0.175, 0.115, "", dashed=True)
    ax.text(0.887, 0.822, r"$\widehat{\tau}_p(0,\mathbf{c})=0$", fontsize=10.5, ha="center")
    ax.text(0.887, 0.790, r"$\partial_\sigma\widehat{\tau}_p\geq0$", fontsize=10.5, ha="center")
    ax.text(0.887, 0.758, r"$\widehat{\tau}_p\geq0$", fontsize=10.5, ha="center")
    arrow(ax, (0.915, y2 + 0.070), (0.887, 0.735), dashed=True, lw=1.1)

    # Side outputs: trusted prior and optional curve head, no long criss-cross arrows.
    block(ax, 0.560, 0.385, 0.105, 0.060, r"$\tau_{\rm BB}$", fs=13)
    block(ax, 0.685, 0.385, 0.095, 0.060, r"$w_{\rm BB}$", fs=13)
    arrow(ax, (0.620, y2), (0.612, 0.445), dashed=True, lw=1.0)
    arrow(ax, (0.665, 0.415), (0.685, 0.415))

    block(ax, 0.400, 0.285, 0.110, 0.060, r"$H_\omega$", fs=15, dashed=True)
    block(ax, 0.540, 0.285, 0.140, 0.060, r"$\widehat{\tau}(x),\widehat{d}(x)$", fs=12, dashed=True)
    arrow(ax, (0.510, 0.315), (0.540, 0.315))
    ax.text(0.455, 0.257, r"$x,\sigma_n,\mathbf{c},\widehat{\tau}_p$", fontsize=9.8, ha="center")

    # Loss row is summary only.
    ax.text(0.060, 0.105, "LOSS =", fontsize=15, ha="center")
    block(ax, 0.145, 0.070, 0.105, 0.070, r"$\mathcal{L}_{data}$", fs=14)
    ax.text(0.267, 0.105, "+", fontsize=16, ha="center")
    block(ax, 0.295, 0.070, 0.095, 0.070, r"$\mathcal{L}_{BB}$", fs=14)
    ax.text(0.407, 0.105, "+", fontsize=16, ha="center")
    block(ax, 0.435, 0.070, 0.105, 0.070, r"$\mathcal{L}_{curv}$", fs=14)
    ax.text(0.557, 0.105, "+", fontsize=16, ha="center")
    block(ax, 0.585, 0.070, 0.115, 0.070, r"$\mathcal{L}_{curve}$", fs=14)
    ax.text(0.850, 0.105, r"$\Theta=\{\phi,\theta,\psi,\omega\}$", fontsize=11, ha="center")
    arrow(ax, (0.700, 0.105), (0.800, 0.105))

    for path in (
        OUT / "geospin_framework_clean_v3.png",
        OUT / "geospin_framework_clean_v3.pdf",
        MANUSCRIPT_FIGURES / "geospin_framework_clean_v3.png",
        MANUSCRIPT_FIGURES / "geospin_framework_clean_v3.pdf",
    ):
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    draw()


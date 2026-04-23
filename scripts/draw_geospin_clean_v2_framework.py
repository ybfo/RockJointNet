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
        color="0.10",
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
        boxstyle="round,pad=0.01,rounding_size=0.014",
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


def label(ax, x, y, text, fs=10.5):
    ax.text(x, y, text, fontsize=fs, ha="center", va="center")


def draw() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.2, 5.4), dpi=270)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Column titles.
    ax.text(0.07, 0.93, "Input", fontsize=13, ha="center")
    ax.text(0.22, 0.93, "Path", fontsize=13, ha="center")
    ax.text(0.44, 0.93, "Latent dynamics", fontsize=13, ha="center")
    ax.text(0.64, 0.93, "Rate head", fontsize=13, ha="center")
    ax.text(0.82, 0.93, "Output / checks", fontsize=13, ha="center")

    # Main two-lane input.
    block(ax, 0.035, 0.725, 0.075, 0.070, r"$\mathbf{c}$", fs=17, bold=True)
    block(ax, 0.035, 0.560, 0.075, 0.070, r"$\sigma$", fs=17, bold=True)

    # Context lane.
    block(ax, 0.155, 0.725, 0.090, 0.070, r"$E_\phi$", fs=16)
    block(ax, 0.290, 0.725, 0.105, 0.070, r"$\mathbf{h}(0)$", fs=15)
    arrow(ax, (0.110, 0.760), (0.155, 0.760))
    arrow(ax, (0.245, 0.760), (0.290, 0.760))

    # Stress/path lane.
    block(ax, 0.155, 0.560, 0.090, 0.070, r"$s$", fs=17)
    block(ax, 0.290, 0.560, 0.105, 0.070, r"$u_k$", fs=17)
    block(ax, 0.440, 0.560, 0.120, 0.070, r"$\sigma(u_k)$", fs=14)
    block(ax, 0.605, 0.560, 0.130, 0.070, r"$\mathbf{q}(u_k,\mathbf{c})$", fs=12)
    arrow(ax, (0.110, 0.595), (0.155, 0.595))
    arrow(ax, (0.245, 0.595), (0.290, 0.595))
    arrow(ax, (0.395, 0.595), (0.440, 0.595))
    arrow(ax, (0.560, 0.595), (0.605, 0.595))
    label(ax, 0.200, 0.675, r"$s=\log(1+\sigma/\sigma_{\rm ref})$")
    label(ax, 0.345, 0.675, r"$u_k=\frac{s}{2}(\xi_k+1)$")
    label(ax, 0.500, 0.675, r"$\sigma_{\rm ref}(e^{u_k}-1)$")
    label(ax, 0.585, 0.635, r"$\mathcal{Q}$", fs=11.5)

    # Latent dynamics and rate head.
    block(ax, 0.440, 0.725, 0.120, 0.070, r"$F_\theta$", fs=16)
    block(ax, 0.605, 0.725, 0.130, 0.070, r"$\mathbf{h}(u_k)$", fs=14)
    arrow(ax, (0.395, 0.760), (0.440, 0.760))
    arrow(ax, (0.560, 0.760), (0.605, 0.760))
    # Short conditioning arrows, no long crossing diagonals.
    arrow(ax, (0.342, 0.630), (0.470, 0.725), dashed=True, rad=-0.03)
    arrow(ax, (0.670, 0.630), (0.530, 0.725), dashed=True, rad=0.03)
    label(ax, 0.500, 0.820, r"$d\mathbf{h}/ds=F_\theta(\mathbf{h},s,\mathbf{c},\mathbf{q})$", fs=10.5)

    block(ax, 0.775, 0.725, 0.105, 0.070, r"$G_\psi$", fs=16)
    block(ax, 0.775, 0.560, 0.105, 0.070, r"$r_\psi$", fs=17)
    arrow(ax, (0.735, 0.760), (0.775, 0.760))
    arrow(ax, (0.735, 0.595), (0.775, 0.595))
    arrow(ax, (0.827, 0.725), (0.827, 0.630))
    label(ax, 0.845, 0.675, r"$\mathrm{softplus}+\varepsilon$", fs=10.5)

    # Integral and final prediction.
    block(ax, 0.905, 0.560, 0.075, 0.070, r"$\widehat{\tau}_p$", fs=17, dashed=True)
    arrow(ax, (0.880, 0.595), (0.905, 0.595))
    label(ax, 0.910, 0.510, r"$\frac{s}{2}\sum_k\omega_k r_\psi(u_k,\mathbf{c})$", fs=10.5)

    # Right-side guarantees, compact and aligned.
    block(ax, 0.815, 0.300, 0.165, 0.145, "", dashed=True)
    ax.text(0.897, 0.420, "checks", fontsize=11, ha="center")
    ax.text(0.897, 0.385, r"$\widehat{\tau}_p(0,\mathbf{c})=0$", fontsize=11, ha="center")
    ax.text(0.897, 0.350, r"$\partial_\sigma\widehat{\tau}_p\geq0$", fontsize=11, ha="center")
    ax.text(0.897, 0.315, r"$\widehat{\tau}_p\geq0$", fontsize=11, ha="center")
    arrow(ax, (0.942, 0.560), (0.920, 0.445), dashed=True)

    # BB prior branch and curve branch are below the main path.
    block(ax, 0.605, 0.360, 0.100, 0.060, r"$\tau_{\rm BB}$", fs=13)
    block(ax, 0.720, 0.360, 0.095, 0.060, r"$w_{\rm BB}$", fs=13)
    arrow(ax, (0.670, 0.560), (0.650, 0.420), dashed=True)
    arrow(ax, (0.705, 0.390), (0.720, 0.390))

    block(ax, 0.435, 0.270, 0.110, 0.060, r"$H_\omega$", fs=15, dashed=True)
    block(ax, 0.575, 0.270, 0.140, 0.060, r"$\widehat{\tau}(x),\widehat{d}(x)$", fs=12, dashed=True)
    label(ax, 0.490, 0.235, r"$x,\sigma_n,\mathbf{c},\widehat{\tau}_p$", fs=10)
    arrow(ax, (0.905, 0.560), (0.545, 0.320), dashed=True, rad=-0.07)
    arrow(ax, (0.545, 0.300), (0.575, 0.300))

    # Loss lane: no criss-crossing, just a clean summary.
    ax.text(0.060, 0.105, "LOSS =", fontsize=15, ha="center")
    block(ax, 0.150, 0.070, 0.105, 0.070, r"$\mathcal{L}_{data}$", fs=14)
    ax.text(0.275, 0.105, "+", fontsize=16, ha="center")
    block(ax, 0.305, 0.070, 0.095, 0.070, r"$\mathcal{L}_{BB}$", fs=14)
    ax.text(0.420, 0.105, "+", fontsize=16, ha="center")
    block(ax, 0.450, 0.070, 0.105, 0.070, r"$\mathcal{L}_{curv}$", fs=14)
    ax.text(0.575, 0.105, "+", fontsize=16, ha="center")
    block(ax, 0.605, 0.070, 0.115, 0.070, r"$\mathcal{L}_{curve}$", fs=14)
    arrow(ax, (0.942, 0.560), (0.205, 0.140), dashed=True, rad=-0.15, lw=1.0)
    arrow(ax, (0.658, 0.360), (0.352, 0.140), dashed=True, rad=0.08, lw=1.0)
    arrow(ax, (0.645, 0.270), (0.662, 0.140), dashed=True, rad=0.0, lw=1.0)
    arrow(ax, (0.897, 0.300), (0.502, 0.140), dashed=True, rad=0.12, lw=1.0)

    # Parameter update shown as small note, not a huge loop.
    ax.text(0.860, 0.105, r"$\Theta=\{\phi,\theta,\psi,\omega\}$", fontsize=11, ha="center")
    arrow(ax, (0.720, 0.105), (0.800, 0.105))

    for path in (
        OUT / "geospin_framework_clean_v2.png",
        OUT / "geospin_framework_clean_v2.pdf",
        MANUSCRIPT_FIGURES / "geospin_framework_clean_v2.png",
        MANUSCRIPT_FIGURES / "geospin_framework_clean_v2.pdf",
    ):
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    draw()


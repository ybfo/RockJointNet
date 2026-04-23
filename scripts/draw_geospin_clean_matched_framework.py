from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
MANUSCRIPT_FIGURES = ROOT.parent / "RockJointNet_TNNLS_manuscript" / "figures"


def arrow(ax, start, end, dashed=False, rad=0.0, lw=1.4):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        color="0.10",
        linewidth=lw,
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(patch)
    return patch


def block(ax, x, y, w, h, text, fs=12, dashed=False, bold=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor="white",
        edgecolor="0.08",
        linewidth=1.25,
        linestyle=(0, (3, 2.5)) if dashed else "solid",
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        weight="bold" if bold else "normal",
    )
    return patch


def draw() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.4, 6.2), dpi=260)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Section labels.
    ax.text(0.065, 0.925, "Input", fontsize=12, ha="center")
    ax.text(0.245, 0.925, "Nondimensionalisation", fontsize=12, ha="center")
    ax.text(0.500, 0.925, "GeoSPIN", fontsize=13, ha="center")
    ax.text(0.855, 0.925, "Physics / AutoDiff", fontsize=12, ha="center")

    # Top lane: static context -> encoder -> latent state.
    block(ax, 0.025, 0.735, 0.09, 0.075, r"$\mathbf{c}$", fs=17, bold=True)
    block(ax, 0.165, 0.735, 0.105, 0.075, r"$E_\phi$", fs=16)
    block(ax, 0.320, 0.735, 0.115, 0.075, r"$\mathbf{h}(0)$", fs=16)
    block(ax, 0.500, 0.735, 0.115, 0.075, r"$F_\theta$", fs=16)
    block(ax, 0.665, 0.735, 0.125, 0.075, r"$\mathbf{h}(u_k)$", fs=15)

    arrow(ax, (0.115, 0.772), (0.165, 0.772))
    arrow(ax, (0.270, 0.772), (0.320, 0.772))
    arrow(ax, (0.435, 0.772), (0.500, 0.772))
    arrow(ax, (0.615, 0.772), (0.665, 0.772))

    # Stress lane: sigma -> s -> nodes -> online physics.
    block(ax, 0.025, 0.555, 0.09, 0.075, r"$\sigma$", fs=17, bold=True)
    block(ax, 0.165, 0.555, 0.105, 0.075, r"$s$", fs=17)
    block(ax, 0.320, 0.555, 0.115, 0.075, r"$u_k$", fs=17)
    block(ax, 0.500, 0.555, 0.115, 0.075, r"$\sigma(u_k)$", fs=14)
    block(ax, 0.665, 0.555, 0.125, 0.075, r"$\mathbf{q}(u_k,\mathbf{c})$", fs=12)

    arrow(ax, (0.115, 0.592), (0.165, 0.592))
    arrow(ax, (0.270, 0.592), (0.320, 0.592))
    arrow(ax, (0.435, 0.592), (0.500, 0.592))
    arrow(ax, (0.615, 0.592), (0.665, 0.592))

    ax.text(0.140, 0.642, r"$s=\log(1+\sigma/\sigma_{\rm ref})$", fontsize=10.5, ha="left")
    ax.text(0.290, 0.642, r"$u_k=\frac{s}{2}(\xi_k+1)$", fontsize=10.5, ha="left")
    ax.text(0.458, 0.642, r"$\sigma_{\rm ref}(e^{u_k}-1)$", fontsize=10.5, ha="left")
    ax.text(0.620, 0.642, r"$\mathcal{Q}$", fontsize=11.5, ha="left")

    # F_theta conditions.
    arrow(ax, (0.377, 0.630), (0.530, 0.735), dashed=True, rad=-0.08)
    arrow(ax, (0.728, 0.630), (0.560, 0.735), dashed=True, rad=0.08)
    arrow(ax, (0.070, 0.735), (0.520, 0.792), dashed=True, rad=-0.08)

    # Rate head lane.
    block(ax, 0.665, 0.385, 0.125, 0.075, r"$G_\psi$", fs=16)
    block(ax, 0.830, 0.385, 0.105, 0.075, r"$r_\psi$", fs=17)
    arrow(ax, (0.728, 0.735), (0.700, 0.460))
    arrow(ax, (0.728, 0.555), (0.700, 0.460))
    arrow(ax, (0.790, 0.422), (0.830, 0.422))
    ax.text(0.805, 0.472, r"$\mathrm{softplus}+\varepsilon$", fontsize=10.5, ha="center")

    # Integral output.
    block(ax, 0.830, 0.225, 0.105, 0.075, r"$\widehat{\tau}_p$", fs=18, dashed=True)
    arrow(ax, (0.882, 0.385), (0.882, 0.300))
    ax.text(0.765, 0.326, r"$\frac{s}{2}\sum_{k=1}^{K}\omega_k r_\psi(u_k,\mathbf{c})$", fontsize=10.8, ha="left")

    # Physics checks.
    block(ax, 0.785, 0.690, 0.180, 0.190, "", dashed=True)
    block(ax, 0.805, 0.805, 0.140, 0.045, r"$\widehat{\tau}_p(0,\mathbf{c})=0$", fs=11)
    block(ax, 0.805, 0.755, 0.140, 0.045, r"$\partial_\sigma\widehat{\tau}_p\geq0$", fs=11)
    block(ax, 0.805, 0.705, 0.140, 0.045, r"$\widehat{\tau}_p\geq0$", fs=11)
    arrow(ax, (0.900, 0.300), (0.875, 0.690), dashed=True, rad=0.08)

    # Trusted BB branch.
    block(ax, 0.500, 0.385, 0.115, 0.075, r"$\tau_{\rm BB}$", fs=14)
    block(ax, 0.500, 0.285, 0.115, 0.075, r"$w_{\rm BB}$", fs=14)
    arrow(ax, (0.728, 0.555), (0.615, 0.422), dashed=True, rad=0.15)
    arrow(ax, (0.728, 0.555), (0.615, 0.322), dashed=True, rad=0.12)

    # Curve head branch.
    block(ax, 0.320, 0.245, 0.115, 0.075, r"$H_\omega$", fs=16, dashed=True)
    block(ax, 0.500, 0.245, 0.145, 0.075, r"$\widehat{\tau}(x),\widehat{d}(x)$", fs=12, dashed=True)
    arrow(ax, (0.882, 0.225), (0.435, 0.285), dashed=True, rad=-0.10)
    arrow(ax, (0.435, 0.282), (0.500, 0.282))
    ax.text(0.382, 0.215, r"$x,\sigma_n,\mathbf{c},\widehat{\tau}_p$", fontsize=10, ha="center")

    # Loss block, deliberately simple and aligned.
    ax.text(0.065, 0.105, "LOSS =", fontsize=15, ha="center")
    block(ax, 0.165, 0.075, 0.105, 0.075, r"$\mathcal{L}_{data}$", fs=14)
    ax.text(0.295, 0.105, "+", fontsize=16, ha="center")
    block(ax, 0.320, 0.075, 0.105, 0.075, r"$\mathcal{L}_{BB}$", fs=14)
    ax.text(0.455, 0.105, "+", fontsize=16, ha="center")
    block(ax, 0.480, 0.075, 0.105, 0.075, r"$\mathcal{L}_{curv}$", fs=14)
    ax.text(0.615, 0.105, "+", fontsize=16, ha="center")
    block(ax, 0.640, 0.075, 0.105, 0.075, r"$\mathcal{L}_{curve}$", fs=14)
    arrow(ax, (0.882, 0.225), (0.220, 0.150), dashed=True, rad=-0.08)
    arrow(ax, (0.557, 0.385), (0.372, 0.150), dashed=True, rad=0.04)
    arrow(ax, (0.557, 0.285), (0.372, 0.150), dashed=True, rad=-0.04)
    arrow(ax, (0.572, 0.245), (0.692, 0.150), dashed=True, rad=-0.05)
    arrow(ax, (0.870, 0.690), (0.532, 0.150), dashed=True, rad=0.08)

    # Update loop.
    ax.text(0.860, 0.105, r"$\Theta=\{\phi,\theta,\psi,\omega\}$", fontsize=11, ha="center")
    arrow(ax, (0.745, 0.105), (0.800, 0.105))
    arrow(ax, (0.830, 0.105), (0.070, 0.555), dashed=True, rad=0.20)

    for path in (
        OUT / "geospin_framework_clean_matched.png",
        OUT / "geospin_framework_clean_matched.pdf",
        MANUSCRIPT_FIGURES / "geospin_framework_clean_matched.png",
        MANUSCRIPT_FIGURES / "geospin_framework_clean_matched.pdf",
    ):
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    draw()


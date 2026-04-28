from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
MANUSCRIPT_FIGURES = ROOT.parent / "RockJointNet_TNNLS_manuscript" / "figures"


def arrow(ax, a, b, lw=1.25, dashed=False, rad=0.0):
    p = FancyArrowPatch(
        a,
        b,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        color="0.12",
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(p)
    return p


def node(ax, x, y, text, r=0.035, fs=13, lw=1.25):
    c = Circle((x, y), r, facecolor="white", edgecolor="0.08", linewidth=lw)
    ax.add_patch(c)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)
    return c


def box(ax, x, y, w, h, text="", fs=12, dashed=False, lw=1.1):
    rect = Rectangle(
        (x, y),
        w,
        h,
        facecolor="white",
        edgecolor="0.08",
        linewidth=lw,
        linestyle=(0, (2.2, 2.2)) if dashed else "solid",
    )
    ax.add_patch(rect)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    return rect


def draw() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.6, 6.3), dpi=260)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Head.
    ax.text(0.05, 0.92, r"$\sigma,\mathbf{c}$", fontsize=17, ha="center")
    ax.text(0.27, 0.90, "NONDIMENSIONALISATION", fontsize=11.5, ha="center")
    ax.text(0.77, 0.91, "AutoDiff", fontsize=11.5, ha="center")

    # Left inputs and nondimensionalization.
    c = node(ax, 0.07, 0.64, r"$\mathbf{c}$", fs=15)
    sig = node(ax, 0.07, 0.48, r"$\sigma$", fs=15)
    s = node(ax, 0.22, 0.48, r"$s$", fs=15)
    h0 = node(ax, 0.22, 0.64, r"$\mathbf{h}(0)$", fs=12)
    arrow(ax, (0.105, 0.64), (0.185, 0.64))
    arrow(ax, (0.105, 0.48), (0.185, 0.48))
    ax.text(0.145, 0.675, r"$E_\phi$", fontsize=11, ha="center")
    ax.text(0.145, 0.515, r"$\log(1+\sigma/\sigma_{\rm ref})$", fontsize=10.5, ha="center")

    # Quadrature and online features.
    uk = node(ax, 0.36, 0.48, r"$u_k$", fs=14)
    sig_u = node(ax, 0.49, 0.48, r"$\sigma(u_k)$", fs=11.5)
    q = node(ax, 0.62, 0.48, r"$\mathbf{q}(u_k,\mathbf{c})$", fs=10.5)
    arrow(ax, (0.255, 0.48), (0.325, 0.48))
    arrow(ax, (0.395, 0.48), (0.455, 0.48))
    arrow(ax, (0.525, 0.48), (0.585, 0.48))
    ax.text(0.29, 0.515, r"$u_k=\frac{s}{2}(\xi_k+1)$", fontsize=10, ha="center")
    ax.text(0.43, 0.515, r"$\sigma_{\rm ref}(e^{u_k}-1)$", fontsize=10, ha="center")
    ax.text(0.555, 0.515, r"$\mathcal{Q}$", fontsize=11, ha="center")

    # Latent dynamics and rate head.
    h = node(ax, 0.42, 0.68, r"$\mathbf{h}(u_k)$", fs=11.5)
    F = node(ax, 0.36, 0.78, r"$F_\theta$", fs=13)
    G = node(ax, 0.62, 0.68, r"$G_\psi$", fs=13)
    r = node(ax, 0.74, 0.58, r"$r_\psi$", fs=14)

    arrow(ax, (0.255, 0.64), (0.385, 0.68))
    arrow(ax, (0.36, 0.515), (0.40, 0.645))
    arrow(ax, (0.62, 0.515), (0.62, 0.645))
    arrow(ax, (0.395, 0.78), (0.42, 0.715))
    arrow(ax, (0.455, 0.68), (0.585, 0.68))
    arrow(ax, (0.655, 0.68), (0.715, 0.60))
    arrow(ax, (0.22, 0.515), (0.33, 0.76), dashed=True, rad=0.15)
    arrow(ax, (0.62, 0.515), (0.39, 0.76), dashed=True, rad=-0.13)
    ax.text(0.39, 0.84, r"$d\mathbf{h}/ds$", fontsize=10.5, ha="center")
    ax.text(0.70, 0.66, r"$\mathrm{softplus}+\varepsilon$", fontsize=10.5, ha="center")

    # Integral output.
    tau = box(ax, 0.80, 0.50, 0.12, 0.12, r"$\widehat{\tau}_p$", fs=18, dashed=True)
    arrow(ax, (0.775, 0.58), (0.80, 0.58))
    ax.text(0.77, 0.52, r"$\frac{s}{2}\sum_k\omega_k r_\psi(u_k,\mathbf{c})$", fontsize=10.5, ha="right")

    # Physics checks.
    box(ax, 0.79, 0.17, 0.16, 0.72, "", dashed=True)
    p1 = node(ax, 0.87, 0.78, r"$\widehat{\tau}_p(0,\mathbf{c})$", fs=10.5, r=0.043)
    p2 = node(ax, 0.87, 0.66, r"$\partial_\sigma\widehat{\tau}_p$", fs=11.5, r=0.043)
    p3 = node(ax, 0.87, 0.34, r"$\widehat{\tau}_p\geq0$", fs=11.5, r=0.043)
    p4 = node(ax, 0.87, 0.23, r"$\partial_s^2\widehat{\tau}_p$", fs=11.5, r=0.043)
    arrow(ax, (0.86, 0.62), (0.87, 0.735))
    arrow(ax, (0.86, 0.62), (0.87, 0.705))
    arrow(ax, (0.86, 0.50), (0.87, 0.385))
    arrow(ax, (0.86, 0.50), (0.87, 0.275))
    ax.text(0.87, 0.12, "Physics\nchecks", fontsize=11.5, ha="center")

    # BB trusted prior branch.
    bb = node(ax, 0.62, 0.28, r"$\tau_{\rm BB}$", fs=11.5)
    wbb = node(ax, 0.74, 0.28, r"$w_{\rm BB}$", fs=11.5)
    arrow(ax, (0.62, 0.445), (0.62, 0.315), dashed=True)
    arrow(ax, (0.66, 0.28), (0.705, 0.28))

    # Optional curve branch.
    curve = box(ax, 0.47, 0.23, 0.12, 0.12, r"$H_\omega$", fs=15, dashed=True)
    yhat = box(ax, 0.47, 0.10, 0.12, 0.08, r"$\widehat{\tau}(x),\widehat{d}(x)$", fs=11, dashed=True)
    arrow(ax, (0.82, 0.50), (0.57, 0.35), lw=1.0)
    arrow(ax, (0.47, 0.285), (0.30, 0.30), lw=1.0, dashed=True)
    arrow(ax, (0.53, 0.23), (0.53, 0.18), lw=1.0)
    ax.text(0.52, 0.215, "curve", fontsize=10, ha="center")

    # Loss block.
    ax.text(0.08, 0.11, "LOSS  =", fontsize=15, ha="center")
    ax.text(0.24, 0.11, r"$\mathcal{L}_{data}$", fontsize=15, ha="center")
    ax.text(0.33, 0.11, "+", fontsize=15, ha="center")
    ax.text(0.41, 0.11, r"$\mathcal{L}_{BB}$", fontsize=15, ha="center")
    ax.text(0.08, 0.055, r"$\mathcal{L}_{curve}$", fontsize=14, ha="center")
    data_box = box(ax, 0.21, 0.18, 0.11, 0.08, "DATA", fs=12, dashed=True)
    arrow(ax, (0.83, 0.50), (0.30, 0.24), lw=1.0)
    arrow(ax, (0.62, 0.245), (0.41, 0.13), lw=1.0)
    arrow(ax, (0.53, 0.10), (0.12, 0.06), lw=1.0)

    # Update loop.
    arrow(ax, (0.30, 0.055), (0.07, 0.055), lw=1.0)
    arrow(ax, (0.07, 0.055), (0.07, 0.445), lw=1.0)

    # Captions inside graph.
    ax.text(0.50, 0.92, "GeoSPIN", fontsize=13, ha="center")
    ax.text(0.62, 0.055, r"$\Theta=\{\phi,\theta,\psi,\omega\}$", fontsize=11, ha="center")

    for path in (
        OUT / "framework_matched.png",
        OUT / "framework_matched.pdf",
        MANUSCRIPT_FIGURES / "framework_matched.png",
        MANUSCRIPT_FIGURES / "framework_matched.pdf",
    ):
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    draw()

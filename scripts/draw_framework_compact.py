from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
MANUSCRIPT_FIGURES = ROOT.parent / "RockJointNet_TNNLS_manuscript" / "figures"


def arrow(ax, a, b, dashed=False, rad=0.0, lw=1.15):
    p = FancyArrowPatch(
        a,
        b,
        arrowstyle="-|>",
        mutation_scale=10.5,
        linewidth=lw,
        color="0.12",
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(p)
    return p


def oval(ax, x, y, text, w=0.075, h=0.055, fs=12.5, lw=1.15):
    e = Ellipse((x, y), w, h, facecolor="white", edgecolor="0.10", linewidth=lw)
    ax.add_patch(e)
    ax.text(x, y, text, fontsize=fs, ha="center", va="center")
    return e


def box(ax, x, y, w, h, text="", fs=11.5, dashed=False, lw=1.05):
    r = Rectangle(
        (x, y),
        w,
        h,
        facecolor="white",
        edgecolor="0.10",
        linewidth=lw,
        linestyle=(0, (2.2, 2.2)) if dashed else "solid",
    )
    ax.add_patch(r)
    if text:
        ax.text(x + w / 2, y + h / 2, text, fontsize=fs, ha="center", va="center")
    return r


def draw() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.8, 7.2), dpi=260)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Titles in the same spirit as the reference PINN graph.
    ax.text(0.055, 0.935, r"$\bar{\sigma},\bar{\mathbf{c}}$", fontsize=17, ha="center")
    ax.text(0.265, 0.900, "NONDIMENSIONALISATION", fontsize=11.5, ha="center")
    ax.text(0.510, 0.900, "CONTEXT / PATH NETWORK", fontsize=11.5, ha="center")
    ax.text(0.835, 0.900, "AutoDiff", fontsize=11.5, ha="center")

    # Raw variables to compact c.
    raw_y = [0.810, 0.770, 0.730, 0.690, 0.650, 0.610]
    raw_labels = ["JRC", "JCS", "UCS", r"$\phi_b$", "L", "E"]
    for y, text in zip(raw_y, raw_labels):
        oval(ax, 0.055, y, text, w=0.055, h=0.030, fs=9.5, lw=0.9)
    c = oval(ax, 0.150, 0.705, r"$\mathbf{c}$", w=0.075, h=0.060, fs=14)
    for y in raw_y:
        arrow(ax, (0.083, y), (0.115, 0.705), lw=0.7)

    # Sigma and s.
    sigma = oval(ax, 0.055, 0.485, r"$\sigma$", w=0.075, h=0.060, fs=15)
    s = oval(ax, 0.150, 0.485, r"$s$", w=0.075, h=0.060, fs=15)
    arrow(ax, (0.092, 0.485), (0.113, 0.485))
    ax.text(0.150, 0.545, r"$\log(1+\sigma/\sigma_{\rm ref})$", fontsize=9.5, ha="center")

    # Context encoding.
    h0 = oval(ax, 0.275, 0.705, r"$\mathbf{h}(0)$", w=0.090, h=0.060, fs=12.5)
    arrow(ax, (0.188, 0.705), (0.230, 0.705))
    ax.text(0.210, 0.735, r"$E_\phi$", fontsize=10.5, ha="center")

    # Quadrature nodes, Q, and online features, like the TXT small-node family.
    uys = [0.585, 0.515, 0.445]
    ulabs = [r"$u_1$", r"$u_2$", r"$u_K$"]
    qlabs = [r"$\mathbf{q}_1$", r"$\mathbf{q}_2$", r"$\mathbf{q}_K$"]
    for y, ul, ql in zip(uys, ulabs, qlabs):
        oval(ax, 0.275, y, ul, w=0.066, h=0.048, fs=11)
        oval(ax, 0.405, y, r"$\mathcal{Q}$", w=0.058, h=0.044, fs=10.5)
        oval(ax, 0.515, y, ql, w=0.066, h=0.048, fs=11)
        arrow(ax, (0.183, 0.485), (0.242, y), lw=0.8)
        arrow(ax, (0.309, y), (0.376, y), lw=0.9)
        arrow(ax, (0.434, y), (0.482, y), lw=0.9)
    ax.text(0.270, 0.635, r"$u_k=\frac{s}{2}(\xi_k+1)$", fontsize=9.5, ha="center")
    ax.text(0.405, 0.635, r"$\sigma(u_k)$", fontsize=9.5, ha="center")
    ax.text(0.515, 0.635, r"$\mathbf{q}(u_k,\mathbf{c})$", fontsize=9.5, ha="center")

    # Latent dynamics and rate head.
    F = oval(ax, 0.405, 0.705, r"$F_\theta$", w=0.078, h=0.052, fs=12.5)
    h = oval(ax, 0.515, 0.705, r"$\mathbf{h}(u_k)$", w=0.095, h=0.055, fs=11)
    G = oval(ax, 0.635, 0.605, r"$G_\psi$", w=0.078, h=0.052, fs=12.5)
    a = oval(ax, 0.735, 0.605, r"$a_k$", w=0.060, h=0.045, fs=11)
    sp = box(ax, 0.782, 0.575, 0.080, 0.060, "softplus", fs=9.8)
    r = oval(ax, 0.905, 0.605, r"$r_\psi$", w=0.068, h=0.050, fs=12.5)

    arrow(ax, (0.320, 0.705), (0.366, 0.705), lw=0.9)
    arrow(ax, (0.444, 0.705), (0.468, 0.705), lw=0.9)
    # F_theta is conditioned by s and q; draw two short dashed links only.
    arrow(ax, (0.275, 0.585), (0.384, 0.680), dashed=True, rad=-0.10, lw=0.75)
    arrow(ax, (0.515, 0.585), (0.420, 0.680), dashed=True, rad=0.10, lw=0.75)
    ax.text(0.420, 0.755, r"$d\mathbf{h}/ds$", fontsize=9.5, ha="center")

    # Concat into G_psi.
    concat = box(ax, 0.570, 0.575, 0.045, 0.060, r"$\oplus$", fs=14)
    arrow(ax, (0.560, 0.690), (0.590, 0.635), lw=0.9)
    arrow(ax, (0.548, 0.585), (0.570, 0.605), lw=0.9)
    arrow(ax, (0.615, 0.605), (0.635, 0.605), lw=0.9)
    arrow(ax, (0.674, 0.605), (0.705, 0.605), lw=0.9)
    arrow(ax, (0.765, 0.605), (0.782, 0.605), lw=0.9)
    arrow(ax, (0.862, 0.605), (0.872, 0.605), lw=0.9)
    ax.text(0.900, 0.655, r"$+\varepsilon$", fontsize=10, ha="center")

    # Integral chain.
    mult = box(ax, 0.785, 0.455, 0.060, 0.050, r"$\times\omega_k$", fs=9.8)
    summ = box(ax, 0.865, 0.455, 0.052, 0.050, r"$\sum_k$", fs=10)
    scale = box(ax, 0.935, 0.455, 0.052, 0.050, r"$\times s/2$", fs=9.2)
    tau = box(ax, 0.885, 0.345, 0.090, 0.062, r"$\widehat{\tau}_p$", fs=14, dashed=True)
    arrow(ax, (0.905, 0.580), (0.815, 0.505), lw=0.9)
    arrow(ax, (0.845, 0.480), (0.865, 0.480), lw=0.9)
    arrow(ax, (0.917, 0.480), (0.935, 0.480), lw=0.9)
    arrow(ax, (0.961, 0.455), (0.935, 0.407), lw=0.9)

    # Physics / AutoDiff panel.
    box(ax, 0.780, 0.140, 0.190, 0.255, "", dashed=True, lw=0.9)
    phys_nodes = [
        (0.875, 0.335, r"$\widehat{\tau}_p(0,\mathbf{c})$"),
        (0.875, 0.275, r"$\partial_\sigma\widehat{\tau}_p$"),
        (0.875, 0.215, r"$\widehat{\tau}_p\geq0$"),
    ]
    for x, y, t in phys_nodes:
        oval(ax, x, y, t, w=0.110, h=0.045, fs=9.8)
    ax.text(0.875, 0.112, "Physics\nchecks", fontsize=10.5, ha="center")
    arrow(ax, (0.930, 0.345), (0.875, 0.358), dashed=True, lw=0.8)

    # BB prior branch within the graph.
    tau_bb = oval(ax, 0.515, 0.365, r"$\tau_{\rm BB}$", w=0.080, h=0.048, fs=11)
    w_bb = oval(ax, 0.635, 0.365, r"$w_{\rm BB}$", w=0.080, h=0.048, fs=11)
    arrow(ax, (0.515, 0.445), (0.515, 0.390), dashed=True, lw=0.75)
    arrow(ax, (0.555, 0.365), (0.595, 0.365), lw=0.9)

    # Optional curve head in dashed box area, visually separate.
    hcur = box(ax, 0.350, 0.250, 0.085, 0.055, r"$H_\omega$", fs=13, dashed=True)
    ycurve = box(ax, 0.465, 0.250, 0.130, 0.055, r"$\widehat{\tau}(x),\widehat{d}(x)$", fs=10.8, dashed=True)
    arrow(ax, (0.435, 0.278), (0.465, 0.278), lw=0.9)
    ax.text(0.392, 0.225, r"$x,\sigma_n,\mathbf{c},\widehat{\tau}_p$", fontsize=9.3, ha="center")

    # Loss block. Keep it close to the reference figure but not connected by messy edges.
    ax.text(0.065, 0.070, "LOSS =", fontsize=13, ha="center")
    ax.text(0.175, 0.070, r"$\mathcal{L}_{data}$", fontsize=13, ha="center")
    ax.text(0.250, 0.070, "+", fontsize=13, ha="center")
    ax.text(0.335, 0.070, r"$\mathcal{L}_{BB}$", fontsize=13, ha="center")
    ax.text(0.410, 0.070, "+", fontsize=13, ha="center")
    ax.text(0.500, 0.070, r"$\mathcal{L}_{curv}$", fontsize=13, ha="center")
    ax.text(0.585, 0.070, "+", fontsize=13, ha="center")
    ax.text(0.680, 0.070, r"$\mathcal{L}_{curve}$", fontsize=13, ha="center")
    ax.text(0.865, 0.070, r"$\Theta=\{\phi,\theta,\psi,\omega\}$", fontsize=10.3, ha="center")

    # Minimal update arrow like the reference, not over the main graph.
    arrow(ax, (0.735, 0.070), (0.810, 0.070), lw=0.9)

    for path in (
        OUT / "framework_compact.png",
        OUT / "framework_compact.pdf",
        MANUSCRIPT_FIGURES / "framework_compact.png",
        MANUSCRIPT_FIGURES / "framework_compact.pdf",
    ):
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    draw()

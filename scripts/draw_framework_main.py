from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
MANUSCRIPT_FIGURES = ROOT.parent / "RockJointNet_TNNLS_manuscript" / "figures"


def arrow(ax, start, end, dashed=False, rad=0.0, lw=1.15):
    p = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=lw,
        color="0.10",
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(p)
    return p


def box(ax, x, y, w, h, text="", fs=9.5, dashed=False, bold=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.010",
        linewidth=1.0,
        edgecolor="0.10",
        facecolor="white",
        linestyle=(0, (3, 2.4)) if dashed else "solid",
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")
    return patch


def rect(ax, x, y, w, h, text="", fs=9.0, dashed=False, bold=False):
    patch = Rectangle(
        (x, y),
        w,
        h,
        linewidth=1.0,
        edgecolor="0.10",
        facecolor="white",
        linestyle=(0, (3, 2.4)) if dashed else "solid",
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")
    return patch


def circle(ax, x, y, r, text, fs=10):
    c = Circle((x, y), r, facecolor="white", edgecolor="0.10", linewidth=1.0)
    ax.add_patch(c)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)
    return c


def draw() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15.5, 8.0), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Left inputs.
    circle(ax, 0.055, 0.835, 0.040, "Queried\nnormal stress\n$\\sigma$", fs=8.5)
    circle(ax, 0.055, 0.700, 0.040, "Static joint\ncontext\n$\\mathbf{c}$", fs=8.5)
    rect(ax, 0.012, 0.615, 0.095, 0.055, "$\\mathbf{c}=[\\mathrm{JRC},\\mathrm{JCS},\\phi_b,$\n$L,\\mathrm{UCS},E,\\mathbf{z}]^\\top$", fs=7.8)

    # Stress transform and features.
    box(
        ax,
        0.135,
        0.785,
        0.140,
        0.115,
        "Stress-path transform\n$s=\\log\\left(1+\\frac{\\sigma}{\\sigma_{\\rm ref}}\\right)$\n$\\sigma(u)=\\sigma_{\\rm ref}(\\exp(u)-1)$",
        fs=8.5,
        bold=True,
    )
    box(ax, 0.165, 0.615, 0.145, 0.070, "Online engineered features\n$\\mathbf{q}(u,\\mathbf{c})=\\mathcal{Q}(\\sigma(u),\\mathbf{c})$", fs=8.5)
    arrow(ax, (0.095, 0.835), (0.135, 0.840))
    arrow(ax, (0.095, 0.700), (0.165, 0.650))
    arrow(ax, (0.205, 0.785), (0.205, 0.685))
    arrow(ax, (0.275, 0.835), (0.345, 0.835))
    ax.text(0.300, 0.853, "$s\\in[0,s_{\\max}]$", fontsize=8.2, ha="center")

    # Main GeoSPIN container.
    box(ax, 0.345, 0.540, 0.440, 0.385, "", fs=9)
    ax.text(0.565, 0.905, "GeoSPIN: constitutive stress-path integral network", fontsize=10.5, weight="bold", ha="center")

    # Encoder and quadrature points.
    box(ax, 0.320, 0.740, 0.070, 0.085, "Context\nencoder\n$\\mathbf{h}(0)=E_{\\phi}(\\mathbf{c})$", fs=7.8)
    arrow(ax, (0.310, 0.650), (0.345, 0.650))
    arrow(ax, (0.240, 0.785), (0.320, 0.785))
    arrow(ax, (0.355, 0.740), (0.355, 0.700))

    ax.text(0.510, 0.805, "Stress-path evolution (quadrature points)", fontsize=8.5, ha="center")
    x_nodes = [0.420, 0.462, 0.500, 0.565, 0.602]
    labels = ["0", "$u_1$", "$u_2$", "$u_K$", "$s$"]
    ax.plot([x_nodes[0], x_nodes[-1]], [0.770, 0.770], color="0.1", lw=1.0)
    for x, lab in zip(x_nodes, labels):
        circle(ax, x, 0.770, 0.006, "")
        ax.text(x, 0.785, lab, fontsize=7.6, ha="center")

    # Dynamics, rate head, and integral.
    box(ax, 0.365, 0.605, 0.145, 0.100, "Latent dynamics\n$\\dfrac{d\\mathbf{h}}{ds}=F_{\\theta}(\\mathbf{h}(s),s,\\mathbf{c},\\mathbf{q}(s,\\mathbf{c}))$", fs=8.0)
    box(ax, 0.535, 0.605, 0.145, 0.100, "Positive rate head\n$r_{\\psi}(u,\\mathbf{c})=$\n$\\mathrm{softplus}(G_{\\psi}(\\mathbf{h}(u),u,\\mathbf{c},\\mathbf{q}(u,\\mathbf{c})))+\\varepsilon$", fs=7.4)
    box(ax, 0.705, 0.580, 0.100, 0.140, "Path-integral\n(quadrature)\n$\\widehat{\\tau}_p(s,\\mathbf{c})=\\int_0^s r(u,\\mathbf{c})du$\n$\\approx\\frac{s}{2}\\sum_{k=1}^K\\omega_k r_{\\psi}(u_k,\\mathbf{c})$", fs=7.4)
    box(ax, 0.565, 0.560, 0.080, 0.038, "$r_{\\psi}\\geq\\varepsilon>0$", fs=8)
    for x in x_nodes[:-1]:
        arrow(ax, (x, 0.764), (x, 0.705), lw=0.8)
    arrow(ax, (0.510, 0.655), (0.535, 0.655))
    arrow(ax, (0.680, 0.655), (0.705, 0.655))
    arrow(ax, (0.607, 0.605), (0.607, 0.598))
    arrow(ax, (0.645, 0.579), (0.705, 0.610), dashed=True, lw=0.8)
    arrow(ax, (0.390, 0.650), (0.365, 0.650))
    arrow(ax, (0.310, 0.650), (0.365, 0.650))

    # Prediction and guarantees.
    circle(ax, 0.850, 0.640, 0.035, "Predicted\npeak shear\nstrength\n$\\widehat{\\tau}_p(s,\\mathbf{c})$", fs=7.7)
    arrow(ax, (0.805, 0.650), (0.815, 0.650), lw=1.15)

    box(
        ax,
        0.885,
        0.745,
        0.105,
        0.200,
        "Constitutive guarantees\n(by construction)\n\n$\\widehat{\\tau}_p(0,\\mathbf{c})=0$\n\n$\\dfrac{\\partial\\widehat{\\tau}_p}{\\partial s}=r_{\\psi}(s,\\mathbf{c})\\geq\\varepsilon$\n\n$\\dfrac{\\partial\\widehat{\\tau}_p}{\\partial\\sigma}=\\dfrac{r_{\\psi}(s,\\mathbf{c})}{\\sigma_{\\rm ref}+\\sigma}>0$",
        fs=7.4,
        dashed=True,
        bold=True,
    )
    arrow(ax, (0.850, 0.675), (0.930, 0.745), dashed=True, lw=0.9)

    # Trusted-region prior.
    box(
        ax,
        0.055,
        0.445,
        0.245,
        0.125,
        "Trusted-region Barton--Bandis prior\n$\\tau_{\\rm BB}(\\sigma,\\mathbf{c})=\\mathcal{B}(\\sigma,\\mathbf{c})$\n$w_{\\rm BB}=\\mathbb{I}[\\sigma\\leq\\sigma_{\\rm trust}]\\,\\mathbb{I}[\\sigma/\\mathrm{JCS}\\leq\\xi]$\nor  $w_{\\rm BB}^{\\rm soft}=\\exp[-\\alpha\\max(0,\\sigma/\\mathrm{JCS}-\\xi)]$",
        fs=7.7,
        dashed=True,
        bold=True,
    )
    box(ax, 0.330, 0.445, 0.055, 0.045, "$\\mathcal{L}_{\\rm BB}$", fs=9)
    arrow(ax, (0.300, 0.505), (0.330, 0.467))
    arrow(ax, (0.385, 0.467), (0.415, 0.405), dashed=True, lw=0.8)

    # Training objective.
    box(ax, 0.065, 0.070, 0.400, 0.220, "", dashed=True)
    ax.text(0.265, 0.260, "Training objective", fontsize=9.5, weight="bold", ha="center")
    box(ax, 0.090, 0.205, 0.145, 0.045, "$\\mathcal{L}_{\\rm data}=\\mathrm{Huber}(\\mathrm{asinh}(\\widehat{\\tau}_p)-\\mathrm{asinh}(\\tau_p))$", fs=7.4)
    box(ax, 0.245, 0.205, 0.115, 0.045, "$\\mathcal{L}_{\\rm curv}=\\mathrm{mean}\\left[(\\partial_s^2\\widehat{\\tau}_p)^2\\right]$", fs=7.4)
    box(ax, 0.380, 0.205, 0.055, 0.045, "$\\mathcal{L}_{\\rm BB}$", fs=8.5)
    box(ax, 0.110, 0.135, 0.310, 0.045, "$\\mathcal{L}=\\lambda_d\\mathcal{L}_{\\rm data}+\\lambda_b\\mathcal{L}_{\\rm BB}+\\lambda_c\\mathcal{L}_{\\rm curv}+\\lambda_w\\|\\Theta\\|_2^2$", fs=8.2)
    box(ax, 0.180, 0.080, 0.170, 0.045, "$\\Theta=\\{\\phi,\\theta,\\psi\\}$\n(Adam update)", fs=8.0)
    arrow(ax, (0.162, 0.205), (0.210, 0.180), lw=0.8)
    arrow(ax, (0.302, 0.205), (0.265, 0.180), lw=0.8)
    arrow(ax, (0.407, 0.205), (0.335, 0.180), lw=0.8)

    # Local curve reconstruction.
    arrow(ax, (0.850, 0.605), (0.850, 0.520))
    box(ax, 0.610, 0.105, 0.355, 0.345, "", fs=8)
    ax.text(0.787, 0.430, "Local direct-shear curve reconstruction", fontsize=9.7, weight="bold", ha="center")
    for x, text in [(0.660, "$x$"), (0.725, "$\\sigma_n$"), (0.790, "$\\mathbf{c}$"), (0.865, "$\\widehat{\\tau}_p(\\sigma_n,\\mathbf{c})$")]:
        circle(ax, x, 0.385, 0.022, text, fs=8)
        arrow(ax, (x, 0.363), (x, 0.335), lw=0.8)
    box(ax, 0.630, 0.285, 0.250, 0.055, "Curve head\n$[\\widehat{\\tau}(x,\\sigma_n,\\mathbf{c}),\\widehat d(x,\\sigma_n,\\mathbf{c})]=xH_{\\omega}(x,\\sigma_n,\\mathbf{c},\\widehat{\\tau}_p)$", fs=7.4)
    box(ax, 0.895, 0.285, 0.065, 0.080, "Zero displacement\n$\\widehat{\\tau}(0)=0$\n$\\widehat d(0)=0$", fs=6.9, dashed=True)
    arrow(ax, (0.895, 0.325), (0.880, 0.312), lw=0.8)
    circle(ax, 0.690, 0.215, 0.030, "Shear\n$\\widehat{\\tau}(x)$", fs=7.2)
    circle(ax, 0.790, 0.215, 0.030, "Dilation\n$\\widehat d(x)$", fs=7.2)
    arrow(ax, (0.705, 0.285), (0.690, 0.245), lw=0.8)
    arrow(ax, (0.790, 0.285), (0.790, 0.245), lw=0.8)
    box(ax, 0.640, 0.135, 0.250, 0.040, "$\\mathcal{L}_{\\rm curve}=\\mathrm{mean}[\\mathrm{Huber}(\\widehat{\\tau}-\\tau)+\\eta\\mathrm{Huber}(\\widehat d-d)]$", fs=7.2)
    box(ax, 0.895, 0.135, 0.065, 0.055, "Compared with\ncalibrated\nC-BB", fs=6.7, dashed=True)
    arrow(ax, (0.890, 0.155), (0.895, 0.162), lw=0.7, dashed=True)

    # Curve objective updates omega, separated from scalar theta.
    arrow(ax, (0.765, 0.135), (0.350, 0.102), dashed=True, lw=0.8)
    ax.text(0.458, 0.092, "$+\\lambda_{\\rm curve}\\mathcal{L}_{\\rm curve}$ for $\\omega$", fontsize=7.5, ha="left")

    for path in (
        OUT / "framework_main.png",
        OUT / "framework_main.pdf",
        MANUSCRIPT_FIGURES / "framework_main.png",
        MANUSCRIPT_FIGURES / "framework_main.pdf",
    ):
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    draw()

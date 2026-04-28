from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
MANUSCRIPT_FIGURES = ROOT.parent / "RockJointNet_TNNLS_manuscript" / "figures"


def add_box(ax, x, y, w, h, title, body=(), fc="#ffffff", ec="#30343b", lw=1.8):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top", fontsize=12.5, weight="bold")
    for i, line in enumerate(body):
        ax.text(x + w / 2, y + h - 0.105 - 0.045 * i, line, ha="center", va="top", fontsize=10.5)
    return box


def add_arrow(ax, start, end, color="#30343b", lw=1.8, rad=0.0, dashed=False):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        linestyle=(0, (4, 4)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    return arrow


def draw() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 6.2), dpi=220)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#fbfaf7")
    ax.set_facecolor("#fbfaf7")

    ax.text(
        0.5,
        0.955,
        "GeoSPIN: Stress-Path Integral Framework",
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
    )
    ax.text(
        0.5,
        0.915,
        "Encode static context once, compute stress-dependent physics online, and integrate a positive neural rate.",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#4b5563",
    )

    # Inputs.
    add_box(
        ax,
        0.035,
        0.58,
        0.17,
        0.23,
        "Static Context",
        (r"$c$: JRC, JCS/UCS, $\phi_b$", "length, E, descriptors"),
        fc="#fff4dc",
        ec="#b7791f",
    )
    add_box(
        ax,
        0.035,
        0.25,
        0.17,
        0.23,
        "Normal Stress Query",
        (r"$\sigma_n$", r"$s=\log(1+\sigma_n/\sigma_{\rm ref})$"),
        fc="#eaf3ff",
        ec="#3b6ea8",
    )

    # Encoder and quadrature.
    add_arrow(ax, (0.205, 0.695), (0.275, 0.695))
    add_box(
        ax,
        0.275,
        0.605,
        0.16,
        0.18,
        "Context Encoder",
        (r"$z_c=E_\phi(c)$", "residual MLP / embeddings"),
        fc="#ffffff",
    )

    add_arrow(ax, (0.205, 0.365), (0.275, 0.365))
    add_box(
        ax,
        0.275,
        0.275,
        0.19,
        0.18,
        "Quadrature Nodes",
        (r"$u_1,u_2,\ldots,u_K \in [0,s]$", r"weights $\omega_k$"),
        fc="#ffffff",
    )
    xs = [0.31, 0.34, 0.37, 0.40, 0.43]
    ax.plot(xs, [0.305] * len(xs), color="#3b6ea8", lw=2.0, alpha=0.4)
    ax.scatter(xs, [0.305] * len(xs), s=34, color="#3b6ea8", zorder=5)

    # Online physics.
    add_arrow(ax, (0.465, 0.365), (0.515, 0.365), color="#2b8a57")
    add_box(
        ax,
        0.515,
        0.26,
        0.18,
        0.23,
        "Online Physics",
        (r"$\sigma_k=\sigma_{\rm ref}(\exp(u_k)-1)$", r"$q_k=Q(u_k,c)$", "BB prior + trust weight"),
        fc="#eaf8f0",
        ec="#2b8a57",
    )

    # Dashed context conditioning to decoder.
    add_arrow(ax, (0.435, 0.695), (0.74, 0.58), color="#64748b", rad=-0.15, dashed=True)
    add_arrow(ax, (0.695, 0.37), (0.74, 0.46), color="#2b8a57")

    # Positive-rate decoder.
    add_box(
        ax,
        0.74,
        0.41,
        0.18,
        0.22,
        "Positive-Rate Decoder",
        (r"$a_k=C_\psi(z_c,u_k,q_k)$", r"$r_k=\mathrm{softplus}(a_k)+\epsilon$", r"$r_k \approx d\hat{\tau}_p/ds$"),
        fc="#fff7e8",
        ec="#b7791f",
    )

    # Integral output.
    add_arrow(ax, (0.92, 0.52), (0.955, 0.52))
    add_box(
        ax,
        0.955,
        0.405,
        0.135,
        0.23,
        "Integral Output",
        (r"$\hat{\tau}_p$", r"$=\int_0^s r(u)\,du$", "MPa"),
        fc="#fff0f0",
        ec="#b33a3a",
    )

    # Guarantees.
    add_box(
        ax,
        0.705,
        0.10,
        0.36,
        0.17,
        "Hard Physical Guarantees",
        (
            r"$\hat{\tau}_p(0,c)=0$     $\hat{\tau}_p\geq0$     $\partial\hat{\tau}_p/\partial\sigma_n\geq0$",
            "guaranteed by positive-rate integration, not batch sorting",
        ),
        fc="#ffffff",
        ec="#2b8a57",
    )
    add_arrow(ax, (1.02, 0.405), (0.975, 0.27), color="#2b8a57", rad=0.25)

    # Training objective strip.
    loss = FancyBboxPatch(
        (0.31, 0.08),
        0.34,
        0.105,
        boxstyle="round,pad=0.018,rounding_size=0.022",
        facecolor="#ffffff",
        edgecolor="#64748b",
        linewidth=1.4,
        linestyle=(0, (4, 4)),
    )
    ax.add_patch(loss)
    ax.text(
        0.48,
        0.145,
        r"Training: Huber(asinh $\hat{\tau}_p$, asinh $\tau_p$) + trusted-region C-BB prior",
        ha="center",
        va="center",
        fontsize=10.2,
    )
    ax.text(
        0.48,
        0.106,
        "C-BB is active only where the empirical prior is trusted.",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#4b5563",
    )
    add_arrow(ax, (0.65, 0.16), (0.77, 0.41), color="#64748b", dashed=True, rad=-0.1)

    # Small stage labels.
    ax.text(0.12, 0.835, "input split", ha="center", fontsize=10, color="#6b7280")
    ax.text(0.355, 0.81, "encode", ha="center", fontsize=10, color="#6b7280")
    ax.text(0.605, 0.515, "node-wise physics", ha="center", fontsize=10, color="#6b7280")
    ax.text(0.83, 0.655, "decode rate", ha="center", fontsize=10, color="#6b7280")
    ax.text(1.02, 0.66, "integrate", ha="center", fontsize=10, color="#6b7280")

    # Use tight bbox so the rightmost output is not clipped.
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"framework_sketch.{ext}", bbox_inches="tight", pad_inches=0.06)
        fig.savefig(MANUSCRIPT_FIGURES / f"framework_sketch.{ext}", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    draw()

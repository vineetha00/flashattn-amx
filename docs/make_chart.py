"""Generate the README speedup chart (light + dark PNGs for GitHub <picture>).

    python docs/make_chart.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = [256, 512, 1024, 2048, 4096, 8192]
SPEEDUP = {  # torch flash ms / ours ms, from bench_results.json (D=64, B=1, H=8)
    False: [1.53, 1.41, 1.77, 1.29, 1.38, 1.29],
    True: [1.75, 2.24, 2.17, 1.67, 1.41, 1.20],
}

MODES = {
    "light": dict(series="#2a78d6", ink="#0b0b0b", secondary="#52514e",
                  muted="#898781", grid="#e1e0d9", baseline="#c3c2b7"),
    "dark": dict(series="#3987e5", ink="#f0efec", secondary="#c3c2b7",
                 muted="#898781", grid="#2c2c2a", baseline="#383835"),
}

for mode, c in MODES.items():
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": c["ink"], "axes.edgecolor": c["baseline"],
        "xtick.color": c["muted"], "ytick.color": c["muted"],
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    for ax, causal, title in zip(axes, (False, True), ("non-causal", "causal")):
        vals = SPEEDUP[causal]
        x = range(len(S))
        ax.bar(x, vals, width=0.55, color=c["series"], zorder=3)
        ax.axhline(1.0, color=c["baseline"], lw=1.2, zorder=2)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.05, f"{v:.2f}×", ha="center", va="bottom",
                    fontsize=9.5, color=c["secondary"])
        ax.set_title(title, fontsize=11, color=c["ink"], pad=8)
        ax.set_xticks(list(x), [str(s) for s in S], fontsize=9)
        ax.set_xlabel("sequence length", fontsize=9.5, color=c["muted"])
        ax.set_ylim(0, 2.6)
        ax.grid(axis="y", color=c["grid"], lw=0.7, zorder=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    axes[0].set_ylabel("speedup  (1× = parity)", fontsize=9.5, color=c["muted"])
    fig.suptitle("flashattn-amx vs PyTorch's flash-attention CPU kernel — "
                 "Apple M3, fp32, D=64", fontsize=12, color=c["ink"], y=1.02)
    fig.tight_layout()
    out = Path(__file__).parent / ("speedup.png" if mode == "light" else "speedup_dark.png")
    fig.savefig(out, dpi=200, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

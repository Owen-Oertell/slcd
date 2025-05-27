#!/usr/bin/env python3
"""
Create a boxen plot comparing sequence-generation algorithms
without background grid lines and with full-length axes.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# ── helpers ──────────────────────────────────────────────────────────────────
def load_npz(
    path: Path, *, key: str = "decoding", baseline_key: str = "baseline"
) -> Tuple[np.ndarray, np.ndarray]:
    """Return `(decoding, baseline)` arrays stored in *path*."""
    with np.load(path) as f:
        return f[key], f[baseline_key]


def block_max(arr: np.ndarray, block: int = 10) -> np.ndarray:
    """Max of every `block`-sized chunk (drops an incomplete tail)."""
    usable = arr[: len(arr) - len(arr) % block]
    return usable.reshape(-1, block).max(axis=1)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    sns.set(style="white", font_scale=1.3)        

    if True:  
        files: Dict[str, str] = {
            "Pre-Trained": "old/log/dna-HepG2.npz",
            "DPS": "old/log/dna-HepG2_DPS.npz",
            "SMC": "old/log/dna-HepG2_TDS.npz",
            "SVDD-PM": "old/log/dna-HepG2_tw.npz",
            "Ours": "log/dna-HepG2-classfier-p2.npz",
        }
    else:
        files: Dict[str, str] = {
            "Pre-Trained": "old/log/rna-MRL.npz",
            "DPS": "old/log/rna-MRL_DPS.npz",
            "SMC": "old/log/rna-MRL_TDS.npz",
            "SVDD-PM": "old/log/rna-MRL_tw.npz",
            "Ours": "log/rna-HepG2_tw.npz",
        }

    # load data ---------------------------------------------------------------
    decodings: Dict[str, np.ndarray] = {}
    baselines: list[np.ndarray] = []

    for label, file in files.items():
        decoding, baseline = load_npz(Path(file))
        decodings[label] = decoding
        baselines.append(baseline)

    pre_trained = np.concatenate(baselines)
    best_n = block_max(pre_trained, block=10)

    # tidy DataFrame ----------------------------------------------------------
    samples = {
        "Pre-Trained": pre_trained,
        "Best-N": best_n,
        "DPS": decodings["DPS"],
        "SMC": decodings["SMC"],
        "SVDD-MC": decodings["Pre-Trained"],  # from dna-HepG2.npz
        "SVDD-PM": decodings["SVDD-PM"],
        "Ours": decodings["Ours"],
    }

    df = pd.DataFrame(
        [(alg, reward) for alg, arr in samples.items() for reward in arr],
        columns=["Algorithm", "Reward"],
    )

    # plot --------------------------------------------------------------------
    palette = {
        "Pre-Trained": "r",
        "Best-N": "b",
        "DPS": "g",
        "SMC": "purple",
        "SVDD-MC": "y",
        "SVDD-PM": "cyan",
        "Ours": "orange",
    }

    g = sns.catplot(
        data=df,
        x="Algorithm",
        y="Reward",
        hue="Algorithm",
        kind="boxen",
        palette=palette,
        height=5,
        aspect=8.2 / 5.0,
    )

    for ax in g.axes.ravel():
        ax.grid(False)

    sns.despine()

    g.fig.tight_layout()
    g.savefig("sequence_dna.png")


if __name__ == "__main__":
    main()

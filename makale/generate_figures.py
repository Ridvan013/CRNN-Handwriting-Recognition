"""
Generate publication-quality figures for the AugCRNN-T paper.

Model naming (must match paper.tex):
  CRNN-S / CRNN-M / CRNN-L  = baseline variants (2/3/4 BiLSTM layers)
  AugCRNN                   = CRNN-L + elastic & morphological augmentation
  AugCRNN-T                 = AugCRNN + trigram post-correction (PROPOSED)

Outputs (in makale/figures/):
  fig1_augmentation_grid.pdf     — 3x4 grid of augmentation examples
  fig2_training_curves.pdf       — train_loss, val_loss, val_WA over 51 epochs
  fig3_confusion_topk.pdf        — top-10 most confused character pairs
  fig4_ablation_bars.pdf         — augmentation ablation bar chart

All figures use:
  - Vector PDF format (scalable, no rasterization)
  - Sans-serif fonts (IEEE compatible)
  - Colorblind-safe palette
  - Consistent style across all figures
"""
from __future__ import annotations
import os
import json
import csv
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

# ---- Paths ---------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

MODEL_DIR = REPO / "Model_aachen_v3_augmented"
HISTORY_JSON = MODEL_DIR / "training_history.json"
TEST_CSV = MODEL_DIR / "test_results_analysis.csv"
IAM_ROOT = REPO / "HTR_Using_CRNN" / "IAM" / "processed" / "archive" / "iam_words" / "words"

# ---- Style ---------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Colorblind-safe palette (Okabe & Ito, adapted)
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_PURPLE = "#CC79A7"
C_YELLOW = "#F0E442"
C_GRAY = "#666666"


# ==========================================================================
# Figure 0: System pipeline diagram
# ==========================================================================
def fig_pipeline():
    """End-to-end system overview: data -> preprocessing -> augmentation ->
    CRNN -> CTC decode -> trigram LM -> evaluation.
    Our two contributions are highlighted (green, bold border)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")
    ax.grid(False)

    BOX_W, BOX_H = 20.0, 12.0
    Y_TOP, Y_BOT = 30.0, 6.0

    def box(x, y, title, lines, fc, ec, lw=1.0, bold_title=True):
        p = FancyBboxPatch((x, y), BOX_W, BOX_H,
                           boxstyle="round,pad=0.5,rounding_size=1.2",
                           linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
        ax.add_patch(p)
        cx, cy = x + BOX_W / 2, y + BOX_H / 2
        ax.text(cx, cy + 3.0, title, ha="center", va="center", fontsize=8,
                fontweight="bold" if bold_title else "normal", zorder=3)
        for i, ln in enumerate(lines):
            ax.text(cx, cy + 0.1 - i * 3.0, ln, ha="center", va="center",
                    fontsize=6.4, color="#333333", zorder=3)
        return (x, y)

    def arrow(x1, y1, x2, y2, style="-|>", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle=style, mutation_scale=11,
                                     linewidth=1.0, color="#444444",
                                     connectionstyle=f"arc3,rad={rad}",
                                     zorder=1))

    GREY_F, GREY_E = "#F2F2F2", "#999999"
    BLUE_F, BLUE_E = "#DCE9F5", C_BLUE
    OURS_F, OURS_E = "#D8F0E6", C_GREEN

    xs = [1.0, 27.0, 53.0, 79.0]

    # --- Top row: data -> preprocessing -> augmentation -> encoder ---------
    box(xs[0], Y_TOP, "IAM Aachen",
        ["writer-disjoint", "31.3K / 1.6K / 5.3K", "word crops"], GREY_F, GREY_E)
    box(xs[1], Y_TOP, "Preprocessing",
        ["grayscale $32{\\times}128$", "invert, scale to $[-1,1]$"], BLUE_F, BLUE_E)
    box(xs[2], Y_TOP, "Augmentation*",
        ["geometric + photometric", "+ elastic + morphological",
         "(training only)"], OURS_F, OURS_E, lw=1.8)
    box(xs[3], Y_TOP, "CRNN encoder",
        ["7-block CNN", "4$\\times$BiLSTM-512", "28.73M params"], BLUE_F, BLUE_E)

    for i in range(3):
        arrow(xs[i] + BOX_W, Y_TOP + BOX_H / 2, xs[i + 1], Y_TOP + BOX_H / 2)

    # --- Wrap arrow: encoder (top-right) down to CTC (bottom-right) --------
    arrow(xs[3] + BOX_W / 2, Y_TOP, xs[3] + BOX_W / 2, Y_BOT + BOX_H)

    # --- Bottom row (right to left): CTC -> trigram -> output -> eval ------
    box(xs[3], Y_BOT, "CTC decoding",
        ["training: CTC loss", "test: greedy decode"], BLUE_F, BLUE_E)
    box(xs[2], Y_BOT, "Trigram LM*",
        ["IAM + NLTK lexicon", "238K types", "edit-distance rescoring"],
        OURS_F, OURS_E, lw=1.8)
    box(xs[1], Y_BOT, "Predicted word",
        ["final transcription"], GREY_F, GREY_E)
    box(xs[0], Y_BOT, "Evaluation",
        ["WA, CER", "Wilson 95\\% CI", "McNemar exact"], GREY_F, GREY_E)

    for i in (3, 2, 1):
        arrow(xs[i], Y_BOT + BOX_H / 2, xs[i - 1] + BOX_W, Y_BOT + BOX_H / 2)

    ax.text(50, 0.6, "* contributions of this work",
            ha="center", va="center", fontsize=6.6, style="italic",
            color=C_GREEN)

    plt.tight_layout()
    out = FIG_DIR / "fig0_pipeline.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"  OK {out.name}")


# ==========================================================================
# Figure 2: Training curves
# ==========================================================================
def fig_training_curves():
    with open(HISTORY_JSON) as f:
        h = json.load(f)
    n_ep = len(h["train_loss"])
    epochs = np.arange(1, n_ep + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # Left: losses
    ax1.plot(epochs, h["train_loss"], color=C_BLUE, lw=1.5, label="Train loss")
    ax1.plot(epochs, h["val_loss"],   color=C_ORANGE, lw=1.5, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CTC loss")
    ax1.set_title("(a) Training / Validation Loss")
    ax1.legend(loc="upper right", frameon=False)
    ax1.set_xlim(0, n_ep + 1)

    # Right: val WA + CER
    ax2.plot(epochs, np.array(h["val_wa"]) * 100, color=C_GREEN, lw=1.5, label="Val WA (%)")
    best_ep = int(np.argmax(h["val_wa"])) + 1
    best_wa = max(h["val_wa"]) * 100
    ax2.axvline(best_ep, color=C_RED, lw=0.8, ls=":", alpha=0.7)
    ax2.scatter([best_ep], [best_wa], color=C_RED, s=30, zorder=5,
                label=f"Best: {best_wa:.2f}% (ep {best_ep})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val WA (%)")
    ax2.set_title("(b) Validation Word Accuracy")
    ax2.legend(loc="lower right", frameon=False)
    ax2.set_xlim(0, n_ep + 1)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    out = FIG_DIR / "fig2_training_curves.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ==========================================================================
# Figure 3: Top-K confused characters
# ==========================================================================
def _levenshtein_pairs(pred: str, true: str) -> list[tuple[str, str]]:
    """Return list of (true_char, pred_char) substitution pairs in DP alignment."""
    m, n = len(true), len(pred)
    if m == 0 or n == 0:
        return []
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if true[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    # Backtrack collecting substitutions only
    i, j = m, n
    pairs = []
    while i > 0 and j > 0:
        if true[i - 1] == pred[j - 1]:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            pairs.append((true[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:
            j -= 1  # insertion
        else:
            i -= 1  # deletion
    return pairs


def fig_confusion_topk():
    pair_counter = Counter()
    with open(TEST_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["correct"] == "1":
                continue
            true = row["ground_truth"]
            pred = row["prediction"]
            # Only compare when substitution-heavy (similar length)
            if abs(len(true) - len(pred)) <= 2 and min(len(true), len(pred)) > 0:
                for tc, pc in _levenshtein_pairs(pred, true):
                    pair_counter[(tc, pc)] += 1

    top10 = pair_counter.most_common(10)
    labels = [f"{tc!r}→{pc!r}" for (tc, pc), _ in top10]
    counts = [c for _, c in top10]

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    bars = ax.barh(range(len(labels)), counts, color=C_BLUE, alpha=0.85, edgecolor="black", lw=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Substitution count (test set)")
    ax.set_title("Top-10 character substitutions (AugCRNN-T)")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "fig3_confusion_topk.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ==========================================================================
# Figure 4: Ablation bar chart
# ==========================================================================
def fig_ablation_bars():
    stages = [
        "V3 baseline aug.",
        "+ wide bright/contr/gamma",
        "+ higher noise (σ=0.05)",
        "+ morphological ops",
        "+ elastic deformation",
    ]
    vals = [82.4, 83.7, 84.2, 86.9, 89.6]  # cumulative val WA
    deltas = [None] + [vals[i] - vals[i - 1] for i in range(1, len(vals))]

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(len(stages))
    colors = [C_GRAY, C_BLUE, C_BLUE, C_ORANGE, C_GREEN]
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="black", lw=0.5)
    ax.set_ylim(75, 95)
    ax.set_ylabel("Val WA (%)")
    ax.set_title("Cumulative augmentation ablation (V3-augmented)")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=15, ha="right")

    for i, (bar, v, d) in enumerate(zip(bars, vals, deltas)):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
        if d is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, v - 1.5,
                    f"+{d:.1f}", ha="center", fontsize=7, color="white", fontweight="bold")

    plt.tight_layout()
    out = FIG_DIR / "fig4_ablation_bars.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ==========================================================================
# Figure 1: Augmentation grid (needs an IAM sample image)
# ==========================================================================
def _find_sample_image() -> Path | None:
    if IAM_ROOT.exists():
        for word_dir in sorted(IAM_ROOT.glob("a01/a01-000u/*.png"))[:5]:
            return word_dir
    return None


def _elastic_deform(img: np.ndarray, alpha: float, sigma_frac: float = 0.08) -> np.ndarray:
    h, w = img.shape
    sigma_px = sigma_frac * max(h, w)
    rng = np.random.default_rng(42)
    dx = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32),
                          (0, 0), sigmaX=sigma_px) * alpha
    dy = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32),
                          (0, 0), sigmaX=sigma_px) * alpha
    xg, yg = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(xg + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(yg + dy, 0, h - 1).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=255)


def fig_augmentation_grid():
    sample = _find_sample_image()
    if sample is None:
        print("  ⚠ IAM sample bulunamadı, augmentation grid atlandı.")
        return
    img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  ⚠ {sample.name} okunamadı.")
        return

    variants = []
    variants.append(("Original", img))

    # Rotation
    h, w = img.shape
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), 7, 1.0)
    variants.append(("Rotation +7°", cv2.warpAffine(img, M_rot, (w, h), borderValue=255)))

    # Shear
    M_shear = np.array([[1, 0.15, 0], [0, 1, 0]], dtype=np.float32)
    variants.append(("Shear", cv2.warpAffine(img, M_shear, (w, h), borderValue=255)))

    # Scale down
    M_sc = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 0.85)
    variants.append(("Scale 0.85×", cv2.warpAffine(img, M_sc, (w, h), borderValue=255)))

    # Elastic deformation
    variants.append(("Elastic (α=4)", _elastic_deform(img, alpha=4.0)))

    # Morphological erosion (thicker text: uses inverted logic since bg=white)
    variants.append(("Erode 2×2", cv2.erode(img, np.ones((2, 2), np.uint8))))

    # Morphological dilation
    variants.append(("Dilate 2×2", cv2.dilate(img, np.ones((2, 2), np.uint8))))

    # Brightness / contrast wide
    bc = np.clip(img.astype(np.float32) * 0.75 + 40, 0, 255).astype(np.uint8)
    variants.append(("Brightness×0.75", bc))

    # Gamma
    gamma = 1.6
    lut = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255).astype(np.uint8)
    variants.append((f"Gamma {gamma}", cv2.LUT(img, lut)))

    # Gaussian noise
    rng = np.random.default_rng(1)
    noisy = np.clip(img.astype(np.float32) + rng.normal(0, 15, img.shape), 0, 255).astype(np.uint8)
    variants.append(("Gauss noise σ=15", noisy))

    # Random erasing
    er = img.copy()
    er[15:25, 40:70] = 255
    variants.append(("Random erasing", er))

    # Compose (elastic + morph)
    compose = _elastic_deform(cv2.erode(img, np.ones((2, 2), np.uint8)), alpha=3.0)
    variants.append(("Elastic+Erode (combo)", compose))

    n = len(variants)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 1.6 * nrows))
    axes = np.array(axes).flatten()
    for i, (name, arr) in enumerate(variants):
        axes[i].imshow(arr, cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(name, fontsize=8)
        axes[i].axis("off")
    for j in range(len(variants), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out = FIG_DIR / "fig1_augmentation_grid.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ==========================================================================
# Main
# ==========================================================================
def main():
    print(f"Repo: {REPO}")
    print(f"Figures: {FIG_DIR}\n")

    print("Generating figures:")
    fig_pipeline()
    fig_training_curves()
    fig_confusion_topk()
    fig_augmentation_grid()
    # NOTE: fig_ablation_bars() disabled — the per-component numbers
    # in that plot were not obtained from real ablation runs. Re-enable
    # only after running each augmentation component in isolation and
    # recording the actual validation WA.

    print(f"\nToplam: {len(list(FIG_DIR.glob('*.pdf')))} PDF üretildi -> {FIG_DIR}")


if __name__ == "__main__":
    main()

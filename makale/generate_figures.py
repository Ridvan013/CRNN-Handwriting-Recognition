"""
Generate publication-quality figures for the CRNN-LX paper.

Model naming (must match paper.tex):
  CRNN-S / CRNN-M / CRNN-L  = baseline variants (2/3/4 BiLSTM layers)
  CRNN-G                    = CRNN-B + elastic & morphological augmentation
  CRNN-LX                   = CRNN-G + frequency-ranked post-correction (PROPOSED)

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

MODEL_DIR = REPO / "Model_abl_full"          # CRNN-LX, tam veri
HISTORY_JSON = MODEL_DIR / "training_history.json"
ABL_MODES = [("none", "no augmentation"),
             ("narrow", "CRNN-B (baseline)"), ("photo", "+ wide photometric"),
             ("elastic", "+ elastic"), ("morph", "+ morphological"),
             ("full", "CRNN-LX (all)")]
# Deterministic single-source evaluation (see cloud/ablation_trigram_all.py):
# the same per-word CRNN-LX predictions (KN trigram + line context) that
# Tables 2-4 and the error analysis use.
TEST_CSV = REPO / "results" / "preds_final" / "preds_full.csv"
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
    """End-to-end system overview: data -> augmentation -> preprocessing ->
    CRNN -> CTC decode -> trigram LM -> evaluation.

    The two stages this paper ablates are outlined; each carries its measured
    effect, so the highlight marks what was studied, not what worked."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    # Drawn at the journal text width (5 in) so LaTeX places it 1:1.
    fig, ax = plt.subplots(figsize=(5.0, 2.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")
    ax.grid(False)

    BOX_W, BOX_H = 21.5, 12.0
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
            ax.text(cx, cy + 0.3 - i * 2.8, ln, ha="center", va="center",
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

    xs = [0.5, 26.0, 51.5, 77.0]

    # --- Top row: data -> preprocessing -> augmentation -> encoder ---------
    box(xs[0], Y_TOP, "IAM Aachen",
        ["writer-disjoint", "48.0K / 7.2K / 20.3K", "word crops"], GREY_F, GREY_E)
    box(xs[1], Y_TOP, "Augmentation*",
        ["affine $+$ photometric", "$+$ elastic, morphological",
         "$64{\\times}256$, training only"], OURS_F, OURS_E, lw=1.8)
    box(xs[2], Y_TOP, "Preprocessing",
        ["invert, scale $[-1,1]$", "resize to $32{\\times}128$"],
        BLUE_F, BLUE_E)
    box(xs[3], Y_TOP, "CRNN encoder",
        ["7-block CNN", "4$\\times$BiLSTM-512", "28.73M params"], BLUE_F, BLUE_E)

    # what the ablations measured, so the highlight cannot be read as a claim
    ax.text(xs[1] + BOX_W / 2, Y_TOP - 1.6,
            'conventional $+2.2$ pp; proposed $+0.2$ pp (n.s., 3 seeds)',
            ha="center", va="top", fontsize=6.0, color=C_GREEN, style="italic")

    for i in range(3):
        arrow(xs[i] + BOX_W, Y_TOP + BOX_H / 2, xs[i + 1], Y_TOP + BOX_H / 2)

    # --- Wrap arrow: encoder (top-right) down to CTC (bottom-right) --------
    arrow(xs[3] + BOX_W / 2, Y_TOP, xs[3] + BOX_W / 2, Y_BOT + BOX_H)

    # --- Bottom row (right to left): CTC -> trigram -> output -> eval ------
    box(xs[3], Y_BOT, "CTC decoding",
        ["training: CTC loss", "test: greedy decode"], BLUE_F, BLUE_E)
    box(xs[2], Y_BOT, "Lexical corrector*",
        ["57K corpus lexicon", r"$\leq$2 edits, KN trigram",
         "own left context"],
        OURS_F, OURS_E, lw=1.8)
    box(xs[1], Y_BOT, "Predicted word",
        ["final transcription"], GREY_F, GREY_E)
    box(xs[0], Y_BOT, "Evaluation",
        ["WA, CER", "Wilson 95% CI", "McNemar exact"], GREY_F, GREY_E)

    for i in (3, 2, 1):
        arrow(xs[i], Y_BOT + BOX_H / 2, xs[i - 1] + BOX_W, Y_BOT + BOX_H / 2)

    ax.text(xs[2] + BOX_W / 2, Y_BOT - 1.4,
            "lexicon $+2.9$; ranking $+1.2$; context $+0.9$ pp",
            ha="center", va="top", fontsize=6.0, color=C_GREEN, style="italic")

    ax.text(50, 0.6, "* stages ablated in this work; italics give their measured effect",
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
    """Validation WA and training loss for the six ablation configurations."""
    import json
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.0, 2.6))
    # One colour per mode (Okabe-Ito, colour-blind safe).  MUST stay the same
    # length as ABL_MODES: zip() would silently drop the trailing modes.
    colors = ["#4d4d4d", "#8c8c8c", "#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    assert len(colors) >= len(ABL_MODES), "add a colour per ablation mode"
    for (mode, label), c in zip(ABL_MODES, colors):
        h = json.load(open(REPO / f"Model_abl_{mode}" / "training_history.json"))
        wa = [v * 100 for v in h["val_wa"]]; ep = range(1, len(wa) + 1)
        lw = 1.8 if mode == "full" else 1.0
        ax1.plot(ep, wa, color=c, lw=lw, label=label)
        best = max(range(len(wa)), key=lambda i: wa[i])
        ax1.plot(best + 1, wa[best], "o", color=c, ms=3.5)
        ax2.plot(ep, h["train_loss"], color=c, lw=lw, label=label)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("validation WA (%)")
    ax1.set_ylim(60, 88); ax1.set_title("(a) validation WA, best epoch marked", fontsize=8)
    ax1.grid(alpha=.3)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("training CTC loss"); ax2.set_yscale("log")
    ax2.set_title("(b) training loss", fontsize=8); ax2.grid(alpha=.3, which="both")
    # one legend for both panels, below them, so it never covers a curve
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=(0, 0.16, 1, 1))
    out = FIG_DIR / "fig2_training_curves.pdf"
    plt.savefig(out)
    plt.close(fig)
    print("written:", out)


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
            # every misrecognized word, same alignment as
            # cloud/paper_stats_trigram.py, so the bars match the text
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
    ax.set_title("Top-10 character substitutions (CRNN-LX)")
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
# A legible multi-letter training crop ("meeting", 339x96 px).  The previous
# version took whatever PNG sorted first in the directory, which was a
# single-letter fragment -- unreadable, and pointless in a figure whose job is
# to show what the transforms do to handwriting.
SAMPLE_REL = "a01/a01-000u/a01-000u-02-06.png"


def _find_sample_image() -> Path | None:
    if not IAM_ROOT.exists():
        return None
    preferred = IAM_ROOT / SAMPLE_REL
    if preferred.exists():
        return preferred
    for cand in sorted(IAM_ROOT.glob("a01/a01-000u/*.png")):
        return cand
    return None


def _elastic_deform(img: np.ndarray, alpha_px: float, sigma_frac: float = 0.08) -> np.ndarray:
    """Elastic deformation with the corrected parametrisation used for
    training: the Gaussian-smoothed field is rescaled to unit RMS, so
    alpha_px is the RMS displacement in pixels (cf. cloud/gpu_aug.py)."""
    rng = np.random.default_rng(3)
    h, w = img.shape
    sigma = sigma_frac * max(h, w)
    def field():
        f = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), sigmaX=sigma)
        return f / max(np.sqrt((f ** 2).mean()), 1e-8) * alpha_px
    dx, dy = field(), field()
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(x + dx, 0, w - 1); map_y = np.clip(y + dy, 0, h - 1)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=255)


def fig_augmentation_grid():
    sample = _find_sample_image()
    if sample is None:
        print("  ! no IAM sample found, the augmentation grid was skipped.")
        return
    img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  ! {sample.name} could not be read.")
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
    variants.append(("Elastic (2 px RMS)", _elastic_deform(img, alpha_px=2.0)))

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
    compose = _elastic_deform(cv2.erode(img, np.ones((2, 2), np.uint8)), alpha_px=2.0)
    variants.append(("Elastic + Erode", compose))

    # Eight panels, one per axis the paper actually discusses: a conventional
    # geometric transform, the three proposed ones (elastic, morphological,
    # wide photometric) and their combination.  The full twelve-panel version
    # spent a third of a column on transforms the ablation shows do not
    # matter, and "Random erasing" was visually indistinguishable anyway.
    keep = ["Original", "Rotation +7\u00b0", "Elastic (2 px RMS)", "Erode 2\u00d72",
            "Dilate 2\u00d72", "Brightness\u00d70.75", "Gauss noise \u03c3=15",
            "Elastic + Erode"]
    order = {k: i for i, k in enumerate(keep)}
    variants = sorted((v for v in variants if v[0] in keep),
                      key=lambda v: order[v[0]])
    assert len(variants) == len(keep), [v[0] for v in variants]

    n = len(variants)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0, 1.15 * nrows))
    axes = np.array(axes).flatten()
    for i, (name, arr) in enumerate(variants):
        axes[i].imshow(arr, cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(name, fontsize=7)
        axes[i].axis("off")
    for j in range(len(variants), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out = FIG_DIR / "fig1_augmentation_grid.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ==========================================================================
# Figure 4: what the post-corrector actually contributes
# ==========================================================================
def fig_lexicon_decomposition():
    """Change in word accuracy against lexicon-free decoding: one line per
    lexicon, one point per corrector (results/ablation_lexicon_source.json).

    The point of the figure is that the lines are further apart than the
    points along them: the lexicon matters more than the ranking rule.
    """
    import json
    src = REPO / "results" / "ablation_lexicon_source.json"
    d = json.load(open(src, encoding="utf-8"))
    g = d["greedy"]["wa_pct"]
    CORR = [("edit distance only", "edit\nonly"),
            ("unigram prior", "$+$unigram\nprior"),
            ("KN3 left-to-right", "$+$KN3,\nleft ctx"),
            ("KN3 whole-line, keep-OOV", "$+$KN3, line,\nkeep-OOV")]
    LEX = [("training (7K)", "training, 7K types (84.8% coverage)", C_BLUE, "o"),
           ("extended (239K)", "word list, 239K (93.9%)", C_ORANGE, "s"),
           ("corpus vocabulary (57K)", "corpus, 57K (96.4%)", C_GREEN, "D")]

    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    x = range(len(CORR))
    ax.axhspan(-3.5, 0, color="#d9d9d9", alpha=0.45, lw=0, zorder=0)
    ax.axhline(0, color="black", lw=0.9, ls="--", zorder=2)
    for key, label, colour, marker in LEX:
        e = d["lexicons"][key]["correctors"]
        y = [e[c]["test"]["wa_pct"] - g for c, _ in CORR]
        ax.plot(list(x), y, marker + "-", color=colour, lw=1.8, ms=5, zorder=3,
                label=label)
    best = d["lexicons"]["corpus vocabulary (57K)"]["correctors"]["KN3 left-to-right"]["test"]
    ax.annotate("CRNN-LX", xy=(2, best["wa_pct"] - g), xytext=(2.05, best["wa_pct"] - g - 1.1),
                fontsize=7.5, color=C_GREEN,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=C_GREEN))
    ax.text(len(CORR) - 0.55, -0.25, "harmful", fontsize=6.5, color="#555555",
            ha="right", va="top")
    ax.text(len(CORR) - 0.55, 0.15, "lexicon-free baseline", fontsize=6.5,
            color="#333333", ha="right", va="bottom")
    ax.set_xticks(list(x))
    ax.set_xticklabels([lab for _, lab in CORR], fontsize=7)
    ax.set_xlim(-0.25, len(CORR) - 0.45)
    ax.set_ylim(-3.5, 6.0)
    ax.set_ylabel("$\\Delta$ word accuracy vs.\nlexicon-free (pp)", fontsize=8)
    ax.grid(alpha=.3, axis="y")
    ax.legend(fontsize=7, frameon=False, loc="upper left", title="lexicon",
              title_fontsize=7)
    plt.tight_layout()
    out = FIG_DIR / "fig4_lexicon_decomposition.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"  OK {out.name}")


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
    fig_lexicon_decomposition()
    fig_augmentation_grid()
    # NOTE: fig_ablation_bars() disabled — the per-component numbers
    # in that plot were not obtained from real ablation runs. Re-enable
    # only after running each augmentation component in isolation and
    # recording the actual validation WA.

    print(f"\nTotal: {len(list(FIG_DIR.glob('*.pdf')))} PDFs written -> {FIG_DIR}")


if __name__ == "__main__":
    main()

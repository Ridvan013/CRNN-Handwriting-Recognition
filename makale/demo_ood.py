"""
Qualitative out-of-distribution demo for the paper.

A handwritten paragraph contributed by a writer who is NOT part of the IAM
database is segmented into words and passed through the current AugCRNN-T
system (optical model + trigram post-correction). The figure produced here
shows the greedy output and the post-corrected output side by side.

Input : static/uploads/temp_page_2_1773734439_test_merged.png  (full-res scan)
Output: makale/figures/fig4_ood_demo.pdf
"""
from __future__ import annotations
import os, sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "cloud"))

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ensemble_inference import CRNN_V3, greedy_decode_from_log_probs
from trigram_lm import TrigramLanguageModel

SRC = REPO / "static" / "uploads" / "temp_page_2_1773734439_test_merged.png"
CKPT = REPO / "Model_aachen_v3_augmented" / "best_model_wa.pth"
LMSRC = REPO / "aachen_splits" / "train_words.txt"
OUT = Path(__file__).resolve().parent / "figures" / "fig4_ood_demo.pdf"

GT = ("Yesterday I walked to the nearby park with my friend to enjoy "
      "the sunshine and relax after our long week").split()


# ------------------------------------------------------- segmentation ----
def segment_words(gray):
    """Line segmentation by horizontal projection, then word segmentation by
    vertical projection with a gap threshold."""
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    rows = (binv > 0).sum(axis=1)
    lines, inside, start = [], False, 0
    for y, v in enumerate(rows):
        if v > 0 and not inside:
            inside, start = True, y
        elif v == 0 and inside:
            inside = False
            if y - start > 12:
                lines.append((start, y))
    if inside:
        lines.append((start, len(rows)))

    words = []
    for (y0, y1) in lines:
        band = binv[y0:y1]
        cols = (band > 0).sum(axis=0)
        gap, run, w0 = 0, False, 0
        segs = []
        for x, v in enumerate(cols):
            if v > 0:
                if not run:
                    run, w0 = True, x
                gap = 0
            elif run:
                gap += 1
                if gap > 26:                      # inter-word gap
                    segs.append((w0, x - gap))
                    run = False
        if run:
            segs.append((w0, len(cols)))
        for (x0, x1) in segs:
            if x1 - x0 < 8:
                continue
            sub = band[:, x0:x1]
            ys = np.where((sub > 0).any(axis=1))[0]
            words.append((x0, y0 + ys[0], x1 - x0, ys[-1] - ys[0] + 1))
    return words


def preprocess(crop):
    t = torch.from_numpy(crop).float() / 255.0
    t = 1.0 - t
    t = (t - 0.5) / 0.5
    t = t.unsqueeze(0).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(32, 128), mode="bilinear",
                                        align_corners=False)
    return t.squeeze(0)


def main():
    img = cv2.imread(str(SRC), cv2.IMREAD_GRAYSCALE)
    boxes = segment_words(img)
    print(f"image {img.shape}, segmented words: {len(boxes)} (reference: {len(GT)})")

    pad = 6
    crops = []
    for (x, y, w, h) in boxes:
        y0, y1 = max(0, y - pad), min(img.shape[0], y + h + pad)
        x0, x1 = max(0, x - pad), min(img.shape[1], x + w + pad)
        crops.append(img[y0:y1, x0:x1])

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN_V3().to(dev)
    model.load_state_dict(torch.load(str(CKPT), map_location=dev))
    model.eval()
    lm = TrigramLanguageModel(str(LMSRC))

    batch = torch.stack([preprocess(c) for c in crops]).to(dev)
    with torch.no_grad():
        if dev.type == "cuda":
            with torch.amp.autocast("cuda"):
                lp = model(batch)
        else:
            lp = model(batch)
    raw = greedy_decode_from_log_probs(torch.log(torch.exp(lp.float()) + 1e-10))
    cor = [lm.correct_word(w) for w in raw]

    n = min(len(GT), len(raw))
    r_ok = sum(raw[i] == GT[i] for i in range(n))
    c_ok = sum(cor[i] == GT[i] for i in range(n))
    print(f"\n{'#':>3}  {'reference':<11} {'greedy':<13} {'+trigram':<13}")
    print("-" * 48)
    for i in range(n):
        f1 = "" if raw[i] == GT[i] else "  <-"
        f2 = "" if cor[i] == GT[i] else "  <-"
        print(f"{i+1:>3}  {GT[i]:<11} {raw[i]:<11}{f1:<4} {cor[i]:<11}{f2}")
    print("-" * 48)
    print(f"greedy    : {r_ok}/{n} = {r_ok/n*100:.1f}%")
    print(f"+ trigram : {c_ok}/{n} = {c_ok/n*100:.1f}%")

    # ------------------------------------------------------------ figure --
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    fig = plt.figure(figsize=(7.0, 3.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.34)

    ax = fig.add_subplot(gs[0])
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 158, 115), 3)
    ys = [b[1] for b in boxes]; xs = [b[0] for b in boxes]
    y2 = max(b[1] + b[3] for b in boxes); x2 = max(b[0] + b[2] for b in boxes)
    ax.imshow(vis[max(0, min(ys) - 40):y2 + 40, max(0, min(xs) - 40):x2 + 40])
    ax.axis("off")
    ax.set_title("(a) Word segmentation of a paragraph written by an unseen writer",
                 fontsize=8.5, pad=3)

    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    ax2.set_title("(b) AugCRNN-T output before and after lexical post-correction",
                  fontsize=8.5, pad=3, loc="left")

    def render_block(y_top, idx):
        """Three aligned rows (reference / greedy / +trigram) for words[idx]."""
        ref = [GT[i] for i in idx]
        rw = [raw[i] for i in idx]
        cr = [cor[i] for i in idx]
        widths = [max(len(a), len(b), len(c)) for a, b, c in zip(ref, rw, cr)]
        xs, x = [], 0.115
        for w in widths:
            xs.append(x)
            x += 0.0116 * (w + 1.6)
        for row, (label, words) in enumerate([("reference", ref),
                                              ("greedy", rw),
                                              ("+trigram", cr)]):
            yy = y_top - row * 0.135
            ax2.text(0.0, yy, label, fontsize=7.2, family="monospace",
                     fontweight="bold" if row else "normal",
                     color="#1a1a1a" if row else "#555555",
                     transform=ax2.transAxes, va="top")
            for xi, wd, rf in zip(xs, words, ref):
                bad = row > 0 and wd != rf
                ax2.text(xi, yy, wd, fontsize=7.2, family="monospace",
                         va="top", transform=ax2.transAxes,
                         color="#D55E00" if bad else "#1a1a1a",
                         fontweight="bold" if bad else "normal")

    render_block(1.00, range(0, 10))
    render_block(0.50, range(10, n))
    ax2.text(0.0, 0.03,
             f"greedy {r_ok}/{n} words correct        "
             f"+trigram {c_ok}/{n} words correct        errors in orange",
             fontsize=7.0, style="italic", color="#555555",
             transform=ax2.transAxes, va="top")

    OUT.parent.mkdir(exist_ok=True)
    plt.savefig(OUT)
    plt.close(fig)
    print(f"\nwritten: {OUT}")
    return raw, cor, r_ok, c_ok, n


if __name__ == "__main__":
    main()

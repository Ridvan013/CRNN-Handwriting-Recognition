#!/usr/bin/env python3
"""
Lexicon / frequency-prior post-processing ablation for all optical models.

Differences to cloud/ablation_lexicon.py (which handles one checkpoint):

  * the Aachen test set and both lexicons are built ONCE and reused, so the
    five models are scored against bit-identical inputs;
  * inference runs in fp32.  The single-model script used autocast/AMP, whose
    fp16 kernels are not bit-reproducible: two runs of the same checkpoint
    disagreed on 2 of 20,310 words (0.01 pp).  fp32 removes that jitter, so
    repeated runs return identical predictions;
  * five post-processing configurations instead of four -- the extended
    lexicon WITHOUT n-gram rescoring is added, which is what isolates the
    contribution of the n-gram prior at the lexicon size actually used.

Configurations (identical optical hypotheses within each model):

    1. none                      raw greedy CTC output
    2. IAM lexicon, edit only    7.2K types, edit distance only
    3. IAM lexicon + n-gram      7.2K types, n-gram rescoring
    4. ext. lexicon, edit only   239K types, edit distance only
    5. ext. lexicon + n-gram     239K types, n-gram rescoring   (= AugCRNN-T)

Rows 2->3 and 4->5 isolate the n-gram prior at each lexicon size; rows 2->4
and 3->5 isolate the lexicon size at each rescoring setting.

Usage:
    python cloud/ablation_lexicon_all.py \
        --iam-words HTR_Using_CRNN/IAM/processed/archive/iam_words/words.txt \
        --iam-root  HTR_Using_CRNN/IAM/processed/archive/iam_words/words

    Seed repeats living in other directories are added as name=dir:
        --modes narrow,narrow_s123=Model_seed_narrow_123,full,full_s123=Model_seed_full_123

Note: the configuration keys written to the JSON ("... + n-gram") are
historical identifiers that figures and scripts depend on; the paper calls
the component a unigram frequency prior (only unigram counts are ever used
on isolated words).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cloud"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_v3 import (
    DEVICE, CRNNModel, CHAR_LIST, PAD_TOKEN,
    IAMDataset, greedy_decode, custom_collate_fn, decode_labels,
)
from trigram_lm import TrigramLanguageModel
from v3_augmented_train import load_iam_aachen
from ablation_lexicon import correct_lexicon_only_fast, score

MODES = ["none", "narrow", "photo", "elastic", "morph", "full"]
LABELS = {"none":   "no augmentation",
          "narrow": "CRNN-B (baseline)",
          "photo":  "+ wide photometric",
          "elastic": "+ elastic",
          "morph":  "+ morphological",
          "full":   "CRNN-LX (all)"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iam-words", type=str, default="")
    p.add_argument("--iam-root", type=str, default="")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--modes", type=str, default=",".join(MODES))
    p.add_argument("--out", type=str, default="results/ablation_lexicon5_all.json")
    p.add_argument("--repeat-check", type=str, default="full",
                   help="re-run this mode a second time to prove determinism")
    p.add_argument("--dump-preds", type=str, default="results/preds_det",
                   help="directory for per-word predictions of the final "
                        "(extended lexicon + n-gram) configuration")
    p.add_argument("--mcnemar-baseline", type=str, default="narrow")
    return p.parse_args()


def hypotheses(model, loader):
    """Greedy CTC hypotheses for the whole loader, fp32 (deterministic).

    fp32 alone makes a pass reproducible *within* a process; across separate
    processes cuDNN may pick a different convolution algorithm and one word
    in 20,310 flipped between two of our runs.  Forcing the deterministic
    algorithms removes that too."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    raw, refs = [], []
    cached_T = None
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            lp = model(images)                      # no autocast -> fp32
            if cached_T is None:
                cached_T = lp.size(0)
            lengths = torch.full((images.size(0),), cached_T,
                                 dtype=torch.long, device=DEVICE)
            for seq in greedy_decode(lp, lengths):
                raw.append(decode_labels(seq))
            for t in labels:
                refs.append("".join(CHAR_LIST[i] for i in t.tolist()
                                    if i != PAD_TOKEN)
                            if isinstance(t, torch.Tensor) else decode_labels(t))
    return raw, refs


def mcnemar_exact(flags_a, flags_b):
    """Exact (binomial) McNemar test on paired per-word correctness flags.

    b = only A correct, c = only B correct.  Under H0 each discordant pair is
    a fair coin, so p = 2 * P(X <= min(b,c)) with X ~ Binomial(b+c, 1/2),
    clipped at 1.
    """
    b = sum(1 for x, y in zip(flags_a, flags_b) if x and not y)
    c = sum(1 for x, y in zip(flags_a, flags_b) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    # cumulative binomial, computed in log space to stay exact for n ~ 1700
    from math import lgamma, exp, log
    logp = -n * log(2.0)
    tail = 0.0
    for i in range(k + 1):
        logc = lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)
        tail += exp(logc + logp)
    return b, c, min(1.0, 2.0 * tail)


def main():
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    t0 = time.time()

    print("=" * 78)
    print(" Lexicon / n-gram ablation, five optical models, fp32 (deterministic)")
    print("=" * 78)

    (_, _, _, _, test_imgs, test_labs) = load_iam_aachen(
        REPO_ROOT,
        iam_words_override=args.iam_words,
        iam_root_override=args.iam_root,
    )
    test_ds = IAMDataset(test_imgs, test_labs, is_training=False)
    loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                        collate_fn=custom_collate_fn)

    aachen_words = str(REPO_ROOT / "aachen_splits" / "train_words.txt")
    print("\n Building IAM-only lexicon ...")
    lm_iam = TrigramLanguageModel(aachen_words, use_nltk_extension=False)
    print("\n Building IAM+NLTK lexicon ...")
    lm_full = TrigramLanguageModel(aachen_words, use_nltk_extension=True)
    memo_iam, memo_full = {}, {}

    out = {"n_samples": len(test_labs),
           "fp32_deterministic": True,
           "lexicon_sizes": {"iam_only": len(lm_iam.vocabulary),
                             "iam_plus_nltk": len(lm_full.vocabulary)},
           "models": {}}

    first_raw, final_flags, final_preds, all_refs = {}, {}, {}, None
    for mode in modes:
        # "name=dir" scores an arbitrary run directory (used for the seed
        # repeats); a bare name means Model_abl_<name>.
        if "=" in mode:
            mode, run_dir = mode.split("=", 1)
            ckpt = REPO_ROOT / run_dir / "best_model_wa.pth"
        else:
            ckpt = REPO_ROOT / f"Model_abl_{mode}" / "best_model_wa.pth"
        if not ckpt.exists():
            print(f" !! missing checkpoint: {ckpt}")
            continue
        model = CRNNModel(img_height=32, img_width=128,
                          num_classes=len(CHAR_LIST) + 1).to(DEVICE)
        model.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
        model.eval()

        raw, refs = hypotheses(model, loader)
        first_raw[mode] = raw
        all_refs = refs

        configs = {
            "none (greedy CTC)":
                raw,
            "IAM lexicon, edit only":
                [correct_lexicon_only_fast(w, lm_iam, memo_iam) for w in raw],
            "IAM lexicon + n-gram":
                [lm_iam.correct_word(w) for w in raw],
            "extended lexicon, edit only":
                [correct_lexicon_only_fast(w, lm_full, memo_full) for w in raw],
            "extended lexicon + n-gram":
                [lm_full.correct_word(w) for w in raw],
        }

        rows = []
        print(f"\n---- {mode}  ({LABELS.get(mode, mode)}) "
              f"[{time.time()-t0:.0f}s] ----")
        print(f"   {'post-correction':<30s}{'WA (%)':>9s}{'CER (%)':>9s}  95% CI")
        for name, preds in configs.items():
            s = score(preds, refs)
            s["name"] = name
            rows.append(s)
            ci = s["wilson_95ci_pct"]
            print(f"   {name:<30s}{s['wa_pct']:>9.2f}{s['cer_pct']:>9.2f}"
                  f"  [{ci[0]:.2f}, {ci[1]:.2f}]")
        final = configs["extended lexicon + n-gram"]
        final_flags[mode] = [p == r for p, r in zip(final, refs)]
        final_preds[mode] = final
        out["models"][mode] = {"label": LABELS.get(mode, mode),
                               "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
                               "configurations": rows}
        del model
        torch.cuda.empty_cache()

    # ── per-word dumps (same evaluation as the tables above) ────────────────
    if args.dump_preds and final_preds:
        import csv
        ddir = Path(args.dump_preds if os.path.isabs(args.dump_preds)
                    else REPO_ROOT / args.dump_preds)
        ddir.mkdir(parents=True, exist_ok=True)
        for mode, preds in final_preds.items():
            with open(ddir / f"preds_{mode}.csv", "w", newline="",
                      encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["idx", "ground_truth", "prediction", "correct"])
                for i, (pr, rf) in enumerate(zip(preds, all_refs)):
                    w.writerow([i, rf, pr, int(pr == rf)])
        print(f"\n per-word predictions written to {ddir}")

    # ── exact McNemar of every mode against the baseline ────────────────────
    base = args.mcnemar_baseline
    if base in final_flags:
        out["mcnemar_vs_" + base] = {}
        print(f"\n exact McNemar against '{base}' (N={len(all_refs):,})")
        print(f"   {'mode':<10s}{'only base':>10s}{'only mode':>10s}"
              f"{'dWA (pp)':>10s}{'p':>10s}")
        for mode, flags in final_flags.items():
            if mode == base:
                continue
            b, c, pv = mcnemar_exact(final_flags[base], flags)
            d = 100.0 * (c - b) / len(flags)
            out["mcnemar_vs_" + base][mode] = {
                "only_baseline_correct": b, "only_mode_correct": c,
                "delta_wa_pp": round(d, 4), "p_value": pv}
            print(f"   {mode:<10s}{b:>10d}{c:>10d}{d:>+10.2f}{pv:>10.3f}")

    # ── determinism proof: score one model twice ────────────────────────────
    rc = args.repeat_check
    if rc in first_raw:
        ckpt = REPO_ROOT / f"Model_abl_{rc}" / "best_model_wa.pth"
        model = CRNNModel(img_height=32, img_width=128,
                          num_classes=len(CHAR_LIST) + 1).to(DEVICE)
        model.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
        model.eval()
        raw2, _ = hypotheses(model, loader)
        ndiff = sum(1 for a, b in zip(first_raw[rc], raw2) if a != b)
        out["determinism_check"] = {"mode": rc, "differing_words": ndiff,
                                    "n": len(raw2)}
        print(f"\n determinism check ({rc}): {ndiff} of {len(raw2)} "
              f"hypotheses differ between two independent passes")

    dst = Path(args.out if os.path.isabs(args.out) else REPO_ROOT / args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n written: {dst}   [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()

"""
Professional Ensemble + TTA Evaluation on Aachen writer-disjoint test set.

Available strategies (auto-enabled based on which model checkpoints exist):
  1.  V3-augmented (raw greedy)                             — single best model
  2.  V3-augmented + Trigram                                — post-hoc LM correction
  3.  V3-augmented + TTA(5 views)                           — inference-time aug
  4.  V3-augmented + TTA + Trigram
  5.  V2 + V3-augmented (softmax avg)                       — 2-model ensemble
  6.  V2 + V3-augmented + Trigram
  7.  Weighted V1 + V2 + V3-augmented (1:2:3)               — 3-model ensemble
  8.  Weighted V1 + V2 + V3-augmented + Trigram
  9.  V3-augmented + WBS (NGrams/beam=50/lm=0.7, IAM+NLTK)  — dict-constrained decode
  10. V3-augmented + WBS + Trigram-fallback                 — WBS with Trigram post-hoc
  11. V3-augmented + TTA + WBS                              — TTA-avg then WBS decode
  12. V2 + V3-augmented + WBS                               — 2-model ensemble → WBS

Statistical validation:
  - Wilson 95% CI per strategy
  - McNemar exact paired test vs baseline strategy (default: "V3-aug + Trigram")

Output:
  - Rich stdout table
  - JSON artifact: results/ensemble_results.json

CLI:
  python ensemble_v2v3_and_tta.py \
      --v1 Model_aachen/best_model_wa.pth \
      --v2 Model_aachen_v2/best_model_wa.pth \
      --v3 Model_aachen_v3_augmented/best_model_wa.pth \
      --output-json results/ensemble_results.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

# Suppress OpenCV C++ core log spam before cv2 import
os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402

from trigram_lm import TrigramLanguageModel
from ensemble_inference import (
    CRNN_V1, CRNN_V2, CRNN_V3,
    preprocess_image, greedy_decode_from_log_probs,
    load_test_samples, wilson_ci,
)

CHAR_LIST = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


# ─────────────────────── CLI ────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v1", type=str,
                   default=str(REPO_ROOT / "Model_aachen" / "best_model_wa.pth"))
    p.add_argument("--v2", type=str,
                   default=str(REPO_ROOT / "Model_aachen_v2" / "best_model_wa.pth"))
    p.add_argument("--v3", type=str,
                   default=str(REPO_ROOT / "Model_aachen_v3_augmented" / "best_model_wa.pth"),
                   help="Prefer V3-augmented (84.54%%). Falls back to Model_aachen_v3.")
    p.add_argument("--iam-root", type=str, default="")
    p.add_argument("--output-json", type=str,
                   default=str(REPO_ROOT / "results" / "ensemble_results.json"))
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--baseline-strategy", type=str, default="V3-aug + Trigram",
                   help="Strategy name to compare all others against via McNemar.")
    p.add_argument("--trigram-corpus", type=str,
                   default=str(REPO_ROOT / "aachen_splits" / "train_words.txt"))
    p.add_argument("--iam-words", type=str,
                   default=str(REPO_ROOT / "HTR_Using_CRNN" / "IAM" / "processed" /
                               "archive" / "iam_words" / "words.txt"),
                   help="IAM words.txt for WBS corpus (IAM ok words).")
    p.add_argument("--wbs-beam", type=int, default=50)
    p.add_argument("--wbs-lm", type=float, default=0.7)
    p.add_argument("--wbs-mode", type=str, default="NGrams", choices=["NGrams", "Words"])
    p.add_argument("--no-wbs", action="store_true",
                   help="Disable WBS strategies even if word-beam-search is installed.")
    return p.parse_args()


# ─────────────────────── Model loading ──────────────────────────────────────

def _load_model(cls, ckpt_path: str, name: str, device):
    """Return model or None if checkpoint missing / load fails."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"  [!] {name}: checkpoint yok ({ckpt_path}) — atlandı.")
        return None
    try:
        m = cls().to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        m.eval()
        print(f"  [✓] {name:5s}: {ckpt_path}")
        return m
    except Exception as e:
        print(f"  [!] {name}: yüklenemedi ({e}) — atlandı.")
        return None


# ─────────────────────── Statistics ─────────────────────────────────────────

def mcnemar_exact(preds_a: list, preds_b: list, targets: list) -> dict:
    """Exact McNemar (binomial) paired test. b = a hits ∧ b misses, c = a misses ∧ b hits."""
    a = [p == t for p, t in zip(preds_a, targets)]
    b = [p == t for p, t in zip(preds_b, targets)]
    B = sum(1 for i in range(len(targets)) if a[i] and not b[i])
    C = sum(1 for i in range(len(targets)) if not a[i] and b[i])
    if B + C > 0:
        chi2 = (abs(B - C) - 1) ** 2 / (B + C)
        n_disc = B + C
        k = min(B, C)
        p = min(1.0, 2 * sum(math.comb(n_disc, i) * (0.5 ** n_disc) for i in range(k + 1)))
    else:
        chi2, p = 0.0, 1.0
    return {"chi2": chi2, "p_exact": p, "b": B, "c": C,
            "significant_p01": bool(p < 0.01)}


# ─────────────────────── Inference core ─────────────────────────────────────

def _forward_softmax(model, batch_tensor):
    """Return softmax probs (T, B, C)."""
    lp = model(batch_tensor)          # log_softmax
    return torch.exp(lp)


def _tta_softmax(model, batch_tensor):
    """Test-Time Augmentation: 5 views (original + 4 augments), averaged softmax."""
    views = [
        batch_tensor,
        TF.affine(batch_tensor, angle=2.0,  translate=[0, 0], scale=1.0,  shear=[0.0, 0.0], fill=-1.0),
        TF.affine(batch_tensor, angle=-2.0, translate=[0, 0], scale=1.0,  shear=[0.0, 0.0], fill=-1.0),
        TF.affine(batch_tensor, angle=0.0,  translate=[0, 0], scale=0.96, shear=[0.0, 0.0], fill=-1.0),
        TF.affine(batch_tensor, angle=0.0,  translate=[0, 0], scale=1.04, shear=[0.0, 0.0], fill=-1.0),
    ]
    accum = None
    for v in views:
        p = torch.exp(model(v))
        accum = p if accum is None else accum + p
    return accum / len(views)


def _decode(probs, eps: float = 1e-10):
    log_probs = torch.log(probs + eps)
    return greedy_decode_from_log_probs(log_probs)


def _trigram_correct(preds: list[str], lm: TrigramLanguageModel) -> list[str]:
    return [lm.correct_word(p) for p in preds]


# ─────────────────────── Word Beam Search ───────────────────────────────────

def _build_wbs_corpus(iam_words_path: str) -> set:
    """WBS corpus = IAM 'ok' words + NLTK English wordlist (~238K vocab)."""
    import re
    alpha_only = re.compile(r"^[A-Za-z]+$")
    words: set[str] = set()
    if iam_words_path and os.path.exists(iam_words_path):
        with open(iam_words_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 9 and parts[1] == "ok":
                    w = "".join(parts[8:])
                    if w and alpha_only.match(w):
                        words.add(w)
    iam_count = len(words)
    try:
        import nltk
        try:
            from nltk.corpus import words as nltk_words
            _ = nltk_words.words()
        except LookupError:
            nltk.download("words", quiet=True)
            from nltk.corpus import words as nltk_words
        added = 0
        for w in nltk_words.words():
            if alpha_only.match(w) and w not in words:
                words.add(w)
                added += 1
        print(f"  WBS corpus: IAM={iam_count:,} + NLTK={added:,} → {len(words):,} total")
    except Exception as e:
        print(f"  ⚠️  NLTK genişletmesi atlandı ({e}); IAM {iam_count:,} kelime.")
    return words


def _init_wbs(iam_words_path: str, mode: str, beam_width: int, lm_smoothing: float):
    """Return (wbs_decoder, mode_used) or (None, None) if unavailable."""
    try:
        from word_beam_search import WordBeamSearch
    except ImportError:
        print("  ⚠️  word-beam-search kurulu değil — WBS stratejileri atlandı.")
        return None, None
    corpus_words = _build_wbs_corpus(iam_words_path)
    if not corpus_words:
        print("  ⚠️  WBS corpus boş — WBS stratejileri atlandı.")
        return None, None
    corpus = " ".join(sorted(corpus_words))
    chars_str = CHAR_LIST  # pip wheel: len(chars)+1 = mat.shape[-1], blank IMPLICIT
    word_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    print(f"  WBS init: mode={mode}, beam={beam_width}, lm_smoothing={lm_smoothing}")
    try:
        wbs = WordBeamSearch(beam_width, mode, lm_smoothing,
                             corpus.encode("utf8"),
                             chars_str.encode("utf8"),
                             word_chars.encode("utf8"))
        return wbs, mode
    except Exception as e:
        print(f"  ⚠️  WBS init hatası ({mode}): {e}")
        if mode == "NGrams":
            print("  Fallback: mode=Words denenecek...")
            try:
                wbs = WordBeamSearch(beam_width, "Words", lm_smoothing,
                                     corpus.encode("utf8"),
                                     chars_str.encode("utf8"),
                                     word_chars.encode("utf8"))
                return wbs, "Words"
            except Exception as e2:
                print(f"  ⚠️  Fallback da başarısız: {e2}")
        return None, None


def _wbs_decode(wbs, probs_torch) -> list[str]:
    """probs_torch: (T, B, C+1) softmax probs → list[str] of length B via WBS."""
    import numpy as np
    mat = probs_torch.detach().cpu().numpy().astype(np.float32)
    decoded_batch = wbs.compute(mat)  # list[list[int]]
    result: list[str] = []
    for d in decoded_batch:
        text = "".join(CHAR_LIST[i] for i in d if 0 <= i < len(CHAR_LIST))
        result.append(text)
    return result


def _wbs_with_trigram_fallback(wbs_preds: list[str], greedy_preds: list[str],
                               lm: TrigramLanguageModel) -> list[str]:
    """If WBS returns empty (highly-OOV image), fall back to Trigram-corrected greedy."""
    out: list[str] = []
    for wp, gp in zip(wbs_preds, greedy_preds):
        if wp.strip():
            out.append(wp)
        else:
            out.append(lm.correct_word(gp))
    return out


# ─────────────────────── Strategies ─────────────────────────────────────────

def evaluate_strategies(models: dict, samples: list, batch_size: int, lm: TrigramLanguageModel,
                        device, wbs=None) -> tuple[dict, list]:
    """
    Run all applicable strategies in a single pass over the test set.
    Returns (strategy_preds, targets) where strategy_preds[name] = list[str].
    wbs: optional WordBeamSearch decoder (None → WBS strategies skipped).
    """
    m3 = models.get("v3")
    m2 = models.get("v2")
    m1 = models.get("v1")

    strategies: dict[str, list[str]] = {}
    def _init(name):
        strategies[name] = []

    _init("V3-aug (raw)")
    _init("V3-aug + Trigram")
    _init("V3-aug + TTA")
    _init("V3-aug + TTA + Trigram")
    if m2 is not None:
        _init("V2 + V3-aug (avg)")
        _init("V2 + V3-aug + Trigram")
    if m1 is not None and m2 is not None:
        _init("Weighted V1+V2+V3-aug (1:2:3)")
        _init("Weighted V1+V2+V3-aug + Trigram")
    if wbs is not None:
        _init("V3-aug + WBS")
        _init("V3-aug + WBS + Trigram-fallback")
        _init("V3-aug + TTA + WBS")
        if m2 is not None:
            _init("V2 + V3-aug + WBS")

    targets: list[str] = []
    print(f"\nRunning inference over {len(samples):,} samples "
          f"(batch={batch_size}) ...")

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            tensors = [preprocess_image(p) for _, _, p in batch]
            keep_idx = [k for k, t in enumerate(tensors) if t is not None]
            if not keep_idx:
                continue
            batch_tensor = torch.stack([tensors[k] for k in keep_idx]).to(device)
            batch_targets = [batch[k][1] for k in keep_idx]

            p3 = _forward_softmax(m3, batch_tensor)
            preds_v3 = _decode(p3)

            # TTA
            p3_tta = _tta_softmax(m3, batch_tensor)
            preds_tta = _decode(p3_tta)

            # 2-model
            if m2 is not None:
                p2 = _forward_softmax(m2, batch_tensor)
                p_v2v3 = (p2 + p3) / 2.0
                preds_v2v3 = _decode(p_v2v3)

            # 3-model weighted 1:2:3
            if m1 is not None and m2 is not None:
                p1 = _forward_softmax(m1, batch_tensor)
                # NOTE: already computed p2 above
                p_w = (p1 * 1 + p2 * 2 + p3 * 3) / 6.0
                preds_w = _decode(p_w)

            # WBS decodes (only if wbs available; skip otherwise for speed)
            if wbs is not None:
                preds_wbs_v3 = _wbs_decode(wbs, p3)
                preds_wbs_tta = _wbs_decode(wbs, p3_tta)
                if m2 is not None:
                    preds_wbs_v2v3 = _wbs_decode(wbs, p_v2v3)

            # Store
            strategies["V3-aug (raw)"].extend(preds_v3)
            strategies["V3-aug + Trigram"].extend(_trigram_correct(preds_v3, lm))
            strategies["V3-aug + TTA"].extend(preds_tta)
            strategies["V3-aug + TTA + Trigram"].extend(_trigram_correct(preds_tta, lm))
            if m2 is not None:
                strategies["V2 + V3-aug (avg)"].extend(preds_v2v3)
                strategies["V2 + V3-aug + Trigram"].extend(_trigram_correct(preds_v2v3, lm))
            if m1 is not None and m2 is not None:
                strategies["Weighted V1+V2+V3-aug (1:2:3)"].extend(preds_w)
                strategies["Weighted V1+V2+V3-aug + Trigram"].extend(_trigram_correct(preds_w, lm))
            if wbs is not None:
                strategies["V3-aug + WBS"].extend(preds_wbs_v3)
                strategies["V3-aug + WBS + Trigram-fallback"].extend(
                    _wbs_with_trigram_fallback(preds_wbs_v3, preds_v3, lm))
                strategies["V3-aug + TTA + WBS"].extend(preds_wbs_tta)
                if m2 is not None:
                    strategies["V2 + V3-aug + WBS"].extend(preds_wbs_v2v3)
            targets.extend(batch_targets)

            processed = min(i + batch_size, len(samples))
            if processed % (batch_size * 20) < batch_size or processed >= len(samples):
                print(f"  {processed:5d}/{len(samples)}")

    return strategies, targets


# ─────────────────────── Reporting ──────────────────────────────────────────

def _fmt_pct(v: float) -> str:
    return f"{v * 100:7.2f}%"


def report(strategies: dict, targets: list, baseline_name: str,
           models_used: dict, output_json: str) -> dict:
    n = len(targets)
    print("\n" + "=" * 78)
    print(f"  ENSEMBLE STRATEGIES ON AACHEN TEST (N={n:,})")
    print("=" * 78)
    print(f"  {'Strategy':<38s} {'WA':>8s} {'Wilson 95% CI':>22s}")
    print(f"  {'-'*38} {'-'*8} {'-'*22}")

    per_strategy = []
    for name, preds in strategies.items():
        correct = sum(1 for p, t in zip(preds, targets) if p == t)
        wa = correct / n
        ci_lo, ci_hi = wilson_ci(correct, n)
        per_strategy.append({
            "name": name, "correct": correct, "wa_pct": round(wa * 100, 4),
            "wilson_95ci_pct": [round(ci_lo * 100, 4), round(ci_hi * 100, 4)],
        })
        print(f"  {name:<38s} {_fmt_pct(wa):>8s}  "
              f"[{ci_lo*100:5.2f}%, {ci_hi*100:5.2f}%]")

    # Baseline for McNemar
    if baseline_name not in strategies:
        # Pick first strategy as fallback
        baseline_name = next(iter(strategies))
    baseline_preds = strategies[baseline_name]

    print(f"\n{'-'*78}")
    print(f"  PAIRED McNemar tests vs baseline: '{baseline_name}'")
    print(f"{'-'*78}")
    print(f"  {'Strategy':<38s} {'Δpp':>7s} {'chi2':>7s} {'p_exact':>12s} {'sig01':>7s}")
    print(f"  {'-'*38} {'-'*7} {'-'*7} {'-'*12} {'-'*7}")

    mcnemar_rows = []
    baseline_correct = sum(1 for p, t in zip(baseline_preds, targets) if p == t)
    for name, preds in strategies.items():
        if name == baseline_name:
            continue
        stats = mcnemar_exact(baseline_preds, preds, targets)
        delta_pp = (sum(1 for p, t in zip(preds, targets) if p == t) - baseline_correct) / n * 100
        row = {
            "strategy": name, "delta_pp": round(delta_pp, 4),
            "chi2": round(stats["chi2"], 4),
            "p_exact": float(f"{stats['p_exact']:.4e}"),
            "b": stats["b"], "c": stats["c"],
            "significant_p01": stats["significant_p01"],
        }
        mcnemar_rows.append(row)
        sig_mark = "YES" if row["significant_p01"] else "no"
        print(f"  {name:<38s} {delta_pp:+7.2f} {row['chi2']:7.2f} "
              f"{row['p_exact']:12.3e} {sig_mark:>7s}")

    # Best strategy summary
    best = max(per_strategy, key=lambda r: r["wa_pct"])
    print("\n" + "=" * 78)
    print(f"  BEST: {best['name']}  →  WA {best['wa_pct']:.2f}% "
          f"CI[{best['wilson_95ci_pct'][0]:.2f}, {best['wilson_95ci_pct'][1]:.2f}]")
    print("=" * 78)

    # Write JSON
    result = {
        "n_samples": n,
        "models_used": models_used,
        "baseline_strategy": baseline_name,
        "strategies": per_strategy,
        "mcnemar_vs_baseline": mcnemar_rows,
        "best_strategy": best,
    }
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Sonuçlar JSON: {out_path}")
    return result


# ─────────────────────── Main ───────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading models (skip missing)...")
    models: dict[str, torch.nn.Module] = {}
    m1 = _load_model(CRNN_V1, args.v1, "V1", device)
    if m1 is not None: models["v1"] = m1
    m2 = _load_model(CRNN_V2, args.v2, "V2", device)
    if m2 is not None: models["v2"] = m2
    # V3 with fallback: prefer v3_augmented, fall back to Model_aachen_v3 if primary missing
    v3_path = args.v3
    if not os.path.exists(v3_path):
        fallback = str(REPO_ROOT / "Model_aachen_v3" / "best_model_wa.pth")
        if os.path.exists(fallback):
            print(f"  V3 augmented yok, fallback: {fallback}")
            v3_path = fallback
    m3 = _load_model(CRNN_V3, v3_path, "V3", device)
    if m3 is None:
        print("HATA: V3 modeli yok — ensemble için zorunlu.")
        sys.exit(1)
    models["v3"] = m3

    if not os.path.exists(args.trigram_corpus):
        print(f"HATA: Trigram corpus yok: {args.trigram_corpus}")
        sys.exit(1)
    print(f"\nTrigram LM yükleniyor: {args.trigram_corpus}")
    lm = TrigramLanguageModel(args.trigram_corpus)

    print("\nLoading Aachen test samples (ok-only)...")
    samples = load_test_samples()
    print(f"  {len(samples):,} sample")

    # Word Beam Search initialization (optional)
    wbs = None
    wbs_mode_used = None
    if not args.no_wbs:
        print("\nWord Beam Search hazırlanıyor...")
        wbs, wbs_mode_used = _init_wbs(args.iam_words, args.wbs_mode,
                                       args.wbs_beam, args.wbs_lm)

    strategies, targets = evaluate_strategies(models, samples, args.batch, lm, device, wbs=wbs)

    models_used = {"v1": args.v1 if m1 is not None else None,
                   "v2": args.v2 if m2 is not None else None,
                   "v3": v3_path,
                   "wbs": {"enabled": wbs is not None,
                           "mode": wbs_mode_used,
                           "beam": args.wbs_beam,
                           "lm_smoothing": args.wbs_lm,
                           "iam_words": args.iam_words if wbs is not None else None}}
    report(strategies, targets, args.baseline_strategy, models_used, args.output_json)


if __name__ == "__main__":
    main()

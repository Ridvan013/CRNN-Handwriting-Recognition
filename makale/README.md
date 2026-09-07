# Paper — AugCRNN-T: A Controlled Study of Augmentation and Lexicon Coverage

**Headline:** **AugCRNN-T** reaches **80.73% word accuracy** (Wilson 95% CI
[80.19%, 81.27%], CER 9.09%) on the **complete** IAM Aachen writer-disjoint
test set (N = 20,310 words, 336 forms, 161 unseen writers). Lexicon-free the
same model gives 78.82% / CER 8.60%.

**Two findings, both measured under identical conditions (same data, network,
schedule, machine, code revision):**

1. **Augmentation contributes nothing measurable.** Five configurations that
   differ only in augmentation reach 80.34–80.73% WA; no pairwise McNemar test
   is significant (p ≥ 0.058). The earlier "+6.48 pp" claim was an artefact of
   a truncated label file (39% of IAM) and mismatched pipelines.
2. **Lexicon coverage decides the sign of post-correction.** Correcting
   against the 7,173-word training vocabulary *hurts* (−2.0 to −2.4 pp on every
   model); extending it with the NLTK word list (239,126 types) *helps*
   (+1.5 to +1.9 pp), while CER rises 8.60 → 9.09.

Plus: the common elastic-deformation parameterisation (normalised blur × α∈[2,5])
displaces pixels by ~0.05 px RMS — a no-op; the paper uses a unit-RMS field × 1–3 px.

## Model naming (fixed — use these names everywhere)

| Name | wide photometric | elastic | morph | test WA |
|---|:---:|:---:|:---:|---:|
| `CRNN-L` (baseline) | no | no | no | 80.35 |
| `+ wide photometric` | yes | no | no | 80.34 |
| `+ elastic` | yes | yes | no | 80.68 |
| `+ morphological` | yes | no | yes | 80.64 |
| **`AugCRNN-T`** (proposed) | yes | yes | yes | **80.73** |
| `AugCRNN` | = AugCRNN-T optical model, greedy CTC, no lexicon | | | 78.82 |

CRNN-S / CRNN-M no longer appear (their numbers came from the truncated data).

## Files

- `paper.tex` — LaTeX source (IEEEtran conference, 7 pages)
- `references.bib` — 20 entries (added: Simard 2003, Wigington 2017)
- `generate_figures.py` — regenerates every figure from `Model_abl_*/`
- `figures/` — 4 vector PDFs

## Figures

| File | Content | Data source |
|---|---|---|
| `fig0_pipeline.pdf` | End-to-end system diagram | drawn |
| `fig1_augmentation_grid.pdf` | 12 transforms on one IAM crop (elastic at 2 px RMS) | real IAM crop + cv2 |
| `fig2_training_curves.pdf` | Validation WA + training loss, 5 configurations | `Model_abl_*/training_history.json` |
| `fig3_confusion_topk.pdf` | Top-10 character substitutions of AugCRNN-T | `Model_abl_full/test_results_analysis.csv` |

```bash
python makale/generate_figures.py
```

## Build

```bash
cd makale && tectonic paper.tex          # or: pdflatex → bibtex → pdflatex ×2
```
Overleaf: upload `makale_overleaf.zip` (repo root), compiler pdfLaTeX.

## Paper structure (7 pages)

1. **Introduction** — HTR/CTC/word-level concepts, 4 contributions
2. **Related Work** — 3 families, "what we adopt, where we differ"
3. **Proposed System** — pipeline, data (47,997 / 7,205 / 20,310), naming, model, augmentation (incl. elastic no-op finding), lexicon post-correction (coverage 84.8% / 94.0%)
4. **Experimental Setup** — WA/CER/Wilson/McNemar (eq. 1–4), reproducibility (local RTX 4070, single seed)
5. **Results** — augmentation ablation (Table II), lexicon ablation (Table III), prior work (Table IV), error analysis
6. **Discussion** — why augmentation doesn't help, coverage as the operative variable, HWRCNet, threats to validity
7. **Conclusion**

## Every number is verifiable

| Claim | Source |
|---|---|
| Table II (5 configs WA/CER/CI) | `results/ablation_lexicon_<mode>.json`, row "AugCRNN-T" |
| Table II McNemar p | per-word flags in `Model_abl_<mode>/test_results_analysis.csv` |
| best val WA (epoch) | `Model_abl_<mode>/training_history.json` |
| Table III (lexicon ablation) | `results/ablation_lexicon_full.json` |
| coverage 84.8% / 94.0% | computed from `aachen_splits/{train,test}_words.txt` + NLTK |
| error analysis counts | `Model_abl_full/test_results_analysis.csv` (Levenshtein alignment) |
| external systems | read from the cited papers |

Full report with all tables: `../results/ABLATION_SONUC.md`.

## Author TODO before submission

- [ ] Fill in the `\author{}` block (currently empty)
- [ ] Re-verify every bib entry against the publisher page
- [ ] One English proofreading pass
- [ ] Decide venue; switch `\documentclass` if needed
- [ ] Optional: multiple seeds per configuration to tighten the augmentation null result
- [ ] Optional: WBS / TTA / ensembling re-evaluated on the full data (removed from the paper; old numbers were from truncated data)

## Repo

https://github.com/Ridvan013/CRNN-Handwriting-Recognition
(branch `feature/aachen-v3-extended-trigram`)

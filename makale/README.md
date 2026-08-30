# Paper — AugCRNN-T (IAM Aachen writer-disjoint, word level)

**Headline:** **AugCRNN-T** achieves **84.54% word accuracy**
(Wilson 95% CI [83.55%, 85.49%], CER 9.21%) on the IAM Aachen
writer-disjoint test set (N=5,338) — **+6.48pp** over the identical
un-augmented CRNN-L baseline (78.06%), McNemar exact p < 10⁻³⁰.

## Model naming (fixed — use these names everywhere)

| Name | BiLSTM | Params | Elastic/morph aug | Trigram | WA |
|---|---|---:|:---:|:---:|---:|
| `CRNN-S` | 2 layers | 8.75M | no | yes | 70.29% |
| `CRNN-M` | 3 layers | 15.46M | no | yes | 72.56% |
| `CRNN-L` | 4 layers | 28.73M | no | yes | 78.06% |
| `AugCRNN` | 4 layers | 28.73M | yes | no | — |
| **`AugCRNN-T`** (proposed) | 4 layers | 28.73M | yes | yes | **84.54%** |

Never write "V3", "V3-augmented" or "our model" in the paper — always
`AugCRNN-T` for the proposed system and `CRNN-S/M/L` for the baselines.

## Files

- `paper.tex` — LaTeX source (IEEEtran conference, 7 pages)
- `references.bib` — 18 entries, all verified against dblp/Springer/IAPR
- `generate_figures.py` — regenerates every figure from the real result files
- `figures/` — 4 vector PDFs (see below)

Ablation experiments (pending) are documented in
[`../cloud/ABLATION_REHBER.md`](../cloud/ABLATION_REHBER.md).

## Figures

| File | Content | Data source |
|---|---|---|
| `fig0_pipeline.pdf` | End-to-end system diagram, contributions highlighted | drawn |
| `fig1_augmentation_grid.pdf` | 12 augmentation transforms on one IAM word | real IAM crop + cv2 |
| `fig2_training_curves.pdf` | Losses + validation WA over 51 epochs | `Model_aachen_v3_augmented/training_history.json` |
| `fig3_confusion_topk.pdf` | Top-10 character substitutions | `Model_aachen_v3_augmented/test_results_analysis.csv` |

Regenerate with:
```bash
python makale/generate_figures.py
```

## Build

Local (Tectonic, no LaTeX install needed — downloads packages on demand):
```bash
cd makale
tectonic paper.tex
```

Classic LaTeX:
```bash
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

Overleaf: upload `makale_overleaf.zip` (repo root) → Compiler: pdfLaTeX → Recompile.

## Paper structure (7 pages)

1. **Introduction** — HTR/CTC/word-level concepts explained, 5 contributions
2. **Related Work** — 3 families (CTC recurrent / attention / lexical decoders), each with "what we adopt, where we differ"
3. **Proposed System** — pipeline figure, naming table, encoder, augmentation, trigram, alternative decoders
4. **Experimental Setup** — WA/CER/Wilson/McNemar formulas (eq. 1–4), reproducibility
5. **Results** — augmentation effect, prior-work comparison, decoder/ensemble table, error analysis
6. **Discussion** — detailed HWRCNet comparison, why simple wins, threats to validity
7. **Conclusion** — short opening sentence, contribution-focused

Pending: two ablation tables (augmentation components, lexicon/trigram stages)
once the Kaggle runs described in `../cloud/ABLATION_REHBER.md` finish.

## Every number in the paper is verifiable

| Claim | Source file |
|---|---|
| CRNN-S 70.29% | `Model_aachen/test_summary_analysis.txt` |
| CRNN-M 72.56% | `Model_aachen_v2/test_summary_analysis.txt` |
| CRNN-L 78.06% | `Model_aachen_v3/test_summary_analysis.txt` |
| **AugCRNN-T 84.54%** | `Model_aachen_v3_augmented/test_results_analysis.csv` (4513/5338) |
| CER values | recomputed from `Character_Errors` / `Word_Length` columns |
| Decoder/ensemble table | `results/ensemble_berhat.json` |
| Training curves | `Model_aachen_v3_augmented/training_history.json` (51 epochs) |
| External baselines | Dutta 2018, Rajesh 2022 (arXiv 2201.00947), Kang 2018 — read from the papers |

No fabricated numbers. The earlier per-component augmentation ablation
was removed because those runs were never actually performed.

## Author TODO before submission

- [ ] Fill in the `\author{}` block in `paper.tex` (currently empty)
- [ ] Re-verify every bib entry against the publisher page
- [ ] One English proofreading pass
- [ ] If a specific venue is chosen, switch `\documentclass` to its template
- [ ] **Run the ablation experiments** — see `../cloud/ABLATION_REHBER.md`
      (Kaggle, ~6.5 h; produces the two tables the supervisor asked for)

## Repo

https://github.com/Ridvan013/CRNN-Handwriting-Recognition
(branch `feature/aachen-v3-extended-trigram`)

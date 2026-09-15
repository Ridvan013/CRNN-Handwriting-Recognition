# Paper — Decomposing Lexicon-Assisted Correction for Word-Level Handwritten Text Recognition on IAM: Lexicon Coverage, Frequency Ranking and Line Context

**Headline:** **CRNN-LX** reaches **81.66% word accuracy** (Wilson 95% CI
[81.13%, 82.19%], CER 8.58%) on the **complete** IAM Aachen writer-disjoint
test set (N = 20,310 words, 336 forms, 161 unseen writers), with a 239K
lexicon and an interpolated Kneser-Ney trigram (IAM + Brown) that uses the
system's **own** outputs for the preceding crops of the line as context.
Without context (unigram prior, the figure for genuinely isolated words)
80.74% / CER 9.09%; lexicon-free 78.83% / CER 8.60%.

## 15 Sept (evening): real trigram with line context

| Post-correction on the CRNN-LX optical model | test WA | CER |
|---|---:|---:|
| none (greedy) | 78.83 | 8.60 |
| 239K lexicon + add-one unigram prior (old corrector) | 80.74 | 9.09 |
| + Kneser-Ney unigram, IAM | 80.74 | 9.09 |
| + KN trigram with line context, IAM | 80.84 | 9.07 |
| + KN unigram, IAM + Brown | 81.32 | 8.73 |
| **+ KN trigram with line context, IAM + Brown (CRNN-LX)** | **81.66** | **8.58** |

- Gain over the unigram prior: +0.84 to +0.95 pp on **all six** optical models
  (`results/ablation_trigram_all.json`); α = 7 selected on validation for each.
- Oracle (reference transcriptions as context, diagnostic only): 81.73.
- Leakage: 0.63% of test 5-grams, 0.11% of 6-grams, 0 of 8-grams occur in
  Brown; longest shared run 7 words, all idioms (`results/brown_leakage.json`).
- Model verified numerically: conditional distributions sum to 1 over the
  271,303-type vocabulary (`cloud/kn_trigram_selftest.py`).
- **Table 2 with this corrector:** CRNN-LX − CRNN-B = +0.44 pp, **p = 0.025**
  (not significant at the paper's p < 0.01, nor after Bonferroni over five
  comparisons); anchor −2.41 pp, p = 8×10⁻²⁷. With the unigram prior the same
  pair was +0.38 pp, p = 0.060. The paper says so explicitly; seed repeats
  decide it.

The sections below describe the state before this change (unigram prior).

Every number in the paper comes from **one deterministic evaluation pass**
(`cloud/ablation_lexicon_all.py`, fp32): five optical models x five
post-corrections against identical inputs. Two independent runs agree on all
20,310 hypotheses. Earlier numbers mixed two evaluation paths (the training
script's own test loop and the lexicon-ablation script) that differed by up to
9 words, and used mixed-precision inference, which is not bit-reproducible.

**Two findings, both measured under identical conditions (same data, network,
schedule, machine, code revision):**

1. **The proposed augmentation contributes nothing measurable — and the
   experiment proves it can see augmentation.** Five configurations that keep
   the conventional pipeline (affine, narrow photometric, noise, gamma, random
   erasing) and differ only in the three proposed transforms reach
   80.33–80.74% WA; no pairwise McNemar test is significant (p ≥ 0.060).
   Switching the conventional pipeline off (`--aug-mode none`, sixth
   configuration) costs **−2.49 pp** (80.36 → 77.87) at **p = 2×10⁻²⁷**, and
   −4.43 pp lexicon-free (78.76 → 74.33). So the null result is **saturation**,
   not an insensitive experiment — the anchor is a positive control.
   The earlier "+6.48 pp" claim was an artefact of a truncated label file
   (39% of IAM) and mismatched pipelines.
2. **Post-correction has two ingredients and they do different jobs.**
   *Coverage* decides the sign: the 7,173-word training vocabulary *hurts*
   (−2.4 to −2.7 pp on every model), the 239,126-type extended list *helps*
   (+0.3 to +0.7 pp). The *frequency prior* that ranks candidates within edit
   distance then supplies most of the magnitude (+1.1 to +1.3 pp) and is worth
   twice as much on the large lexicon as on the small one, because a larger
   candidate set makes the ranking matter more. Together: +1.5 to +1.9 pp,
   while CER rises 8.60 → 9.09.

Plus: the common elastic-deformation parameterisation (normalised blur × α∈[2,5])
displaces pixels by only 0.02–0.04 px RMS per axis at the 64×256 working
resolution (max 0.19 px) — a no-op; the paper uses a unit-RMS field × 1–3 px.

## Model naming (fixed — use these names everywhere)

| Name | wide photometric | elastic | morph | test WA |
|---|:---:|:---:|:---:|---:|
| `no augmentation` (anchor) | — conventional pipeline off too — | | | 77.87 |
| `CRNN-B` (baseline) | no | no | no | 80.36 |
| `+ wide photometric` | yes | no | no | 80.33 |
| `+ elastic` | yes | yes | no | 80.67 |
| `+ morphological` | yes | no | yes | 80.64 |
| **`CRNN-LX`** (proposed) | yes | yes | yes | **80.74** |
| `CRNN-G` | = CRNN-LX optical model, greedy CTC, no lexicon | | | 78.83 |

CRNN-S / CRNN-M no longer appear (their numbers came from the truncated data).

## Files

- `paper.tex` — LaTeX source, **IJPRAI journal format** (`ws-ijprai`, single
  column, 18 pages)
- `paper_ieee.tex` — the earlier IEEEtran conference version (two column,
  8 pages), kept for reference. It compiles on its own. Its title is kept in
  sync, but its body predates the 15 Sept revision below.
- `ws-ijprai.cls`, `ws-ijprai.bst` — World Scientific class and bibliography
  style, from the publisher's `ijprai-2e` package
- `references.bib` — 21 entries (added: Simard 2003, Wigington 2017, Mondal 2022)
- `generate_figures.py` — regenerates every figure from `Model_abl_*/`
- `figures/` — 4 vector PDFs

## Figures

| File | Fig. | Content | Data source |
|---|---|---|---|
| `fig0_pipeline.pdf` | 1 | End-to-end system diagram; the two ablated stages are starred | drawn |
| `fig1_augmentation_grid.pdf` | 2 | 8 transforms on one IAM crop ("meeting"), elastic at 2 px RMS | real IAM crop + cv2 |
| `fig2_training_curves.pdf` | 3 | Validation WA + training loss, **6** configurations | `Model_abl_*/training_history.json` |
| `fig4_lexicon_decomposition.pdf` | 4 | **Headline finding**: ΔWA vs lexicon size, with and without the unigram frequency prior | `results/ablation_lexicon5_all.json` |
| `fig3_confusion_topk.pdf` | 5 | Top-10 character substitutions of CRNN-LX | `results/preds_det/preds_full.csv` |

Three defects were fixed in this pass: the pipeline box still carried the
**truncated** split (31.3K/1.6K/5.3K instead of 48.0K/7.2K/20.3K) and marked
augmentation as a contribution; the augmentation grid used whatever PNG sorted
first on disk, an illegible single-letter fragment; and `colors` had five
entries after `ABL_MODES` grew to six, so `zip()` silently dropped the proposed
system from its own training-curve figure.

```bash
python makale/generate_figures.py
```

## Build

```bash
cd makale && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```
Overleaf: upload `makale_overleaf.zip` (repo root), compiler pdfLaTeX. The zip
carries `ws-ijprai.cls` and `ws-ijprai.bst`, so no template needs installing.

### What the IJPRAI conversion changed

Content is unchanged; only what the journal mandates was touched.

| | IEEEtran (`paper_ieee.tex`) | IJPRAI (`paper.tex`) |
|---|---|---|
| Layout | two column, 8 pages | single column, 18 pages |
| Authors | `\IEEEauthorblockN/A` | one `\author`+`\address` pair each |
| Front matter | — | `\markboth`, `\catchline`, `\history` |
| Abstract | 306 words, contained math | **199 words**, no math, no citations |
| Keywords | `IEEEkeywords` | `\keywords{a; b; c.}` |
| Lists | `itemize` | `itemlist` |
| Tables | `\caption` + `\resizebox` | `\tbl{caption\label{}}{body}`, `\colrule`/`\botrule`, `tabfootnote` |
| Figures | `\caption` | `+ \alttext{}`, widths in cm |
| Citations | `[1]` bracketed, citation order | superscript (`overcite`), **alphabetical** |
| Spelling | British | **American** (44 words changed) |

Two latent `references.bib` defects surfaced under the stricter style and were
fixed: `RWTH Aachen University` printed as the person "R. A. University"
(needed double braces) and `Arthur Flôr de Sousa Neto` lost its surname
particle (needed `{de Sousa Neto}` as the family name).

## Paper structure (8 pages)

1. **Introduction** — HTR/CTC/word-level concepts, 4 contributions
2. **Related Work** — 4 groups, "what we adopt, where we differ"
3. **Proposed System** — pipeline, data (47,997 / 7,205 / 20,310), naming, model, augmentation (incl. elastic no-op finding), lexicon post-correction (coverage 84.8% / 94.0%)
4. **Experimental Setup** — WA/CER/Wilson/McNemar (eq. 1–4), reproducibility (local RTX 4070, single seed)
5. **Results** — augmentation ablation (Table II), lexicon ablation (Table III), prior work with a lexicon column (Table IV), error analysis
6. **Discussion** — why augmentation doesn't help, coverage vs. the frequency prior, same-protocol comparison with Sueiras et al. 2018 (+2.6 pp lexicon-free, +4.5 pp with lexicon), HWRCNet note, threats to validity
7. **Conclusion**

## Every number is verifiable

| Claim | Source |
|---|---|
| Table II (5 configs WA/CER/CI) | `results/ablation_lexicon5_all.json`, row "extended lexicon + n-gram" |
| Table II McNemar p | `results/ablation_lexicon5_all.json`, key `mcnemar_vs_narrow` (exact binomial on `results/preds_det/preds_<mode>.csv`) |
| best val WA (epoch) | `Model_abl_<mode>/training_history.json` |
| anchor row (−2.49 pp, p = 2×10⁻²⁷) | `Model_abl_none/`, scored on the same deterministic path |
| Table III (post-correction ablation) | `results/ablation_lexicon5_all.json`, model `full` |
| coverage 84.8% / 94.0% | computed from `aachen_splits/{train,test}_words.txt` + NLTK |
| error analysis counts, Fig. 4 | `results/preds_det/preds_full.csv` (Levenshtein alignment) |
| external systems | read from the cited papers; Sueiras 2018 (WER 23.8 / CER 8.8, lexicon-free) cross-checked in Dutta 2018 Tab. III, Kang 2021 Tab. 7, Kass & Vats 2022 Tab. 5, Mondal 2022 Tab. 1; the "Dutta 77.14" row is HWRCNet's own re-training (Rajesh 2022 Tab. 2), Dutta's own figure is 12.61 % WER |

Full report with all tables: `../results/ABLATION_SONUC.md`.


## Revision of 15 September 2026 (advisor's review)

| # | Advisor's point | Change |
|---|---|---|
| 1 | §6.1 heading asserted a conclusion | → *Interpretation of the Augmentation Results* |
| 2 | §6.3 heading named one paper | → *Comparison with Previous Word-Level HTR Systems* |
| 3 | Title too assertive | → *Decomposing Lexicon-Assisted Correction for Isolated Handwritten Word Recognition on IAM: Effects of Lexicon Coverage and Frequency-Based Ranking* |
| 4 | "trigram" without word context | The corrector is a **unigram frequency prior** everywhere (`score_word` is always called with `prev_words=None`, so bigram/trigram branches never run). §3.5 says so explicitly and adds the sparsity figure: only 34 % of test tokens have a trigram context seen in the 48 K-word training text. |
| 5 | Single seed | (a) seed repeats: see `cloud/ABLATION_REHBER.md` §10 — **pending**, Berhat runs them on Kaggle; (b) wording softened to the advisor's phrasing in the abstract, contribution 2, §5.1, §6.1, §6.3 and the Conclusion |
| 6 | Table 2 ΔWA ≠ numbers in text | Table 2 now has Δ_B (vs CRNN-B) **and** Δ_P (vs +wide photometric); caption states both come from unrounded accuracies |
| 7 | "2–4 pp" | → "1.8–3.9 pp below the later attention-based systems listed in Table 4" |

Three further corrections found while checking Section 3 against the code:
morphological perturbation fires with probability **0.15**, not 0.3 (half of
the 0.3 draws pick a 1×1 no-op kernel); early stopping resets on an
improvement of validation loss **or** WA; the legacy elastic amplitude is
0.02–0.04 px RMS per axis (max 0.19 px), measured with the released code at
64×256.

## Author TODO before submission

- [x] `\author{}` block filled: Nur Banu Oğur (corresponding, nbogur@sakarya.edu.tr), Rıdvan Dursun, Berhat Yeşilyurt; Dept. of Software Engineering, Faculty of Computer and Information Sciences, Sakarya University
- [ ] Confirm Berhat's institutional e-mail (derived from the `ad.soyad@ogr.sakarya.edu.tr` pattern, not verified)
- [ ] Re-verify every bib entry against the publisher page
- [ ] One English proofreading pass
- [ ] Decide venue; switch `\documentclass` if needed
- [ ] **Seed repeats** (advisor, 5a): CRNN-B and CRNN-LX with seeds 123 and 456 — Berhat, on Kaggle, per `cloud/ABLATION_REHBER.md` §10; then a mean ± SD row for the paper
- [ ] Optional: WBS / TTA / ensembling re-evaluated on the full data (removed from the paper; old numbers were from truncated data)

## Repo

https://github.com/Ridvan013/CRNN-Handwriting-Recognition
(branch `feature/aachen-v3-extended-trigram`)

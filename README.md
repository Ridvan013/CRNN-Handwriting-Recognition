# Decomposing Lexicon-Assisted Correction for Word-Level Handwritten Text Recognition on IAM

Code, data lists and per-word predictions for the paper

> R. Dursun, B. Yeşilyurt and N. B. Oğur Etçioğlu, *Decomposing Lexicon-Assisted
> Correction for Word-Level Handwritten Text Recognition on IAM: Lexicon
> Coverage, Frequency Ranking and Line Context.*

The paper source and the compiled PDF are in [`makale/`](makale/).

## What the paper reports

A plain CRNN trained from scratch on the IAM Aachen writer-disjoint split,
followed by a lexical post-corrector, reaches **83.80 % word accuracy**
(7.96 % CER, Wilson 95 % CI [83.29, 84.30]) on **all 20,310 test words**.
The paper takes that corrector apart and measures each part separately:

| Ingredient | Effect on word accuracy |
|---|---|
| lexicon read off the language-model corpus, instead of a 239 K English word list | **+2.14 pp** |
| ranking the candidates by a frequency prior instead of edit distance | +1.19 pp |
| Kneser–Ney trigram with the system's own line context instead of the prior | +0.89 pp |
| imposing the same lexicon inside the decoder (word beam search) instead of after it | +0.32 pp (n.s.) |

and finds, with a positive control and three seeds, that the three
writing-aware augmentations under study (elastic, morphological, widened
photometric) change accuracy by no more than 0.48 pp, with the sign of the
difference flipping between seeds — while removing the conventional
augmentation pipeline beneath them costs 2.24 pp.

## Layout

| Path | Contents |
|---|---|
| [`makale/`](makale/) | the paper (LaTeX + PDF), its figures, and a README describing how every number is produced |
| [`aachen_splits/`](aachen_splits/) | the word-level split lists actually used, the form→writer map, and the official RWTH uttlists |
| [`cloud/`](cloud/) | the model, the training script, the GPU augmentation, the Kneser–Ney trigram and correctors, and every ablation and verification script |
| [`results/`](results/) | one JSON per experiment plus the per-word predictions behind every table |
| `verify_aachen_splits.py` | 25 checks on the partition (writer, form and prompt disjointness, record and image integrity) |
| `trigram_lm.py` | the add-one unigram corrector used as the no-context reference |

Trained weights are not in the repository (they exceed the file-size limit);
they are available from the corresponding author on request. The training
histories that the paper's figures and tables use *are* included, under
`Model_abl_*/training_history.json`.

## Reproducing the paper's numbers

No GPU is needed for any of these: they read the per-word predictions that
are in the repository.

```bash
python verify_aachen_splits.py            # 25 checks on the split itself
python cloud/verify_paper_numbers.py      # every table cell, macro and quoted count
python cloud/check_paper_consistency.py   # labels, citations, and a sweep over every % figure
python cloud/kn_trigram_selftest.py       # the trigram sums to one; back-off and determinism
python cloud/viterbi_selftest.py          # the whole-line decoder equals exhaustive search
python cloud/count_oov_hypotheses.py      # the out-of-lexicon counts of Section 3.5
python cloud/writer_bootstrap.py          # the writer-level bootstrap of Section 6.4
```

`verify_paper_numbers.py` compares the paper against the result files and
exits non-zero on any mismatch, so it can gate a commit.

Re-running an experiment from the checkpoints instead (GPU, and the IAM crops
under `--iam-root`):

```bash
python cloud/ablation_final.py            # Table 2: the six augmentation configurations
python cloud/ablation_lexicon_source.py   # Table 4: three lexicons x four correctors
python cloud/ablation_wbs.py              # Table 5: word beam search on the same probabilities
python cloud/ablation_keep_oov.py         # Section 5.2: the whole-line decoder, split in two
```

## Data

IAM is not redistributed here. The word crops come from the IAM Handwriting
Database (U.-V. Marti and H. Bunke, 2002) and the partition from the RWTH
Aachen writer-disjoint split; `aachen_splits/` holds only the identifier
lists and the transcriptions that the split files themselves contain.

The external text resources of the language model — the NLTK English word
list and the Brown corpus — are downloaded by the scripts on first use.

## Earlier work in this repository

The repository also holds an earlier line of work that the paper does not
use: a page-level demo pipeline with CRAFT text detection (`pipeline_v2.py`,
`greedy*.py`) and a small web front end (`app.py`, `frontend/`). Those parts
are kept for the project's history; their comments and notes are in Turkish,
and the large assets they need (detector weights, uploaded samples, the demo
database) are not tracked.

## Requirements

Python 3.11+, PyTorch 2.x, torchvision, OpenCV, NumPy, NLTK, matplotlib.
The word beam search baseline additionally needs the reference implementation
`word_beam_search` (Scheidl et al.), built from source.

## License

MIT

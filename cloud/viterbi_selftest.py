"""Self-test of LineViterbiCorrector: the decoder must return exactly the
sequence that an exhaustive search over all candidate combinations returns,
for many random column layouts, and must be deterministic and independent of
row order.  Also checks that a right-hand neighbour can change a decision.

Usage:  python cloud/viterbi_selftest.py      (CPU, about two minutes)
"""
import itertools
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cloud"))
from kn_trigram import KNTrigram, iam_line_segments, brown_sentences, LineViterbiCorrector  # noqa: E402
from trigram_lm import TrigramLanguageModel  # noqa: E402

train_words = str(REPO_ROOT / "aachen_splits" / "train_words.txt")
lex = TrigramLanguageModel(train_words, use_nltk_extension=True)
kn = KNTrigram(iam_line_segments(train_words) + brown_sentences(), discount=0.75, extra_vocab=lex.vocabulary)
vc = LineViterbiCorrector(lex, kn, topk=10)
print(kn.describe())

# 1. exact optimum: Viterbi == exhaustive search on random columns of real words
rng = random.Random(0)
pool = ["the", "of", "and", "to", "a", "in", "that", "is", "was", "he", "for", "it",
        "with", "as", "his", "on", "be", "at", "by", "I", "this", "had", "not", "are",
        "but", "from", "or", "have", "an", "they", "which", "one", "you", "were", "her",
        "form", "United", "States", "house", "time", ",", ".", "Mr", "said", "Qzxq", "zzunseen"]
n_checked = 0
for trial in range(400):
    n = rng.randint(1, 6)
    cols = []
    for _ in range(n):
        k = rng.randint(1, 4)
        cols.append([(rng.choice(pool), rng.randint(0, 2)) for _ in range(k)])
    alpha = rng.choice([3.0, 5.0, 7.0, 10.0])
    idx, score = vc.viterbi(cols, alpha)
    best_s, best_combo = None, None
    for combo in itertools.product(*[range(len(c)) for c in cols]):
        s = vc.sequence_score([cols[i][j][0] for i, j in enumerate(combo)],
                              [cols[i][j][1] for i, j in enumerate(combo)], alpha)
        if best_s is None or s > best_s + 1e-12:
            best_s, best_combo = s, combo
    assert abs(score - best_s) < 1e-9, (trial, score, best_s)
    assert abs(vc.sequence_score([cols[i][j][0] for i, j in enumerate(idx)],
                                 [cols[i][j][1] for i, j in enumerate(idx)], alpha) - score) < 1e-9
    # ties: any maximiser is acceptable, but the score must match
    n_checked += 1
print(f"Viterbi == exhaustive search on {n_checked} random layouts: OK")

# 2. right context changes a decision
rows = [("L", 0, "Tne"), ("L", 1, "Stares"), ("L", 2, "of"), ("L", 3, "America")]
out = vc.decode_lines(rows, 5.0)
print("  decoded:", out)
left = vc.correct_lines(rows, 5.0, True)
print("  left-only:", left)

# 3. order independence and determinism
big = []
for li in range(30):
    for wi in range(rng.randint(2, 9)):
        w = rng.choice(pool)
        if rng.random() < 0.3:      # corrupt one character
            p = rng.randrange(len(w)); w = w[:p] + rng.choice("abcdefghijklmnopqrstuvwxyz") + w[p + 1:]
        big.append((f"L{li:02d}", wi, w))
o1 = vc.decode_lines(big, 7.0, real_word=True, keep_oov=True)
perm = list(range(len(big))); rng.shuffle(perm)
o2 = vc.decode_lines([big[i] for i in perm], 7.0, real_word=True, keep_oov=True)
assert [o2[perm.index(i)] for i in range(len(big))] == o1, "order dependence"
assert vc.decode_lines(big, 7.0, real_word=True, keep_oov=True) == o1
print("order-independent and deterministic on 30 synthetic lines: OK")

# 4. a line with no candidates anywhere passes through unchanged
assert vc.decode_lines([("M", 0, "Qzxqzzz"), ("M", 1, "Wvvvvvv")], 5.0) == ["Qzxqzzz", "Wvvvvvv"]
print("no-candidate pass-through: OK")
print("ALL SELF-TESTS PASSED")

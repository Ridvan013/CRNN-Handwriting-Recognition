#!/usr/bin/env python3
"""
Interpolated Kneser-Ney trigram language model + line-context post-corrector.

Why this exists
---------------
The paper's post-corrector ranks lexicon candidates with a unigram frequency
prior: ``TrigramLanguageModel.correct_word`` is always called without previous
words, so its bigram/trigram branches never run.  This module is the *real*
thing: a properly smoothed trigram (Chen & Goodman's interpolated Kneser-Ney)
that scores each candidate given the words that precede it on the same text
line.  The context is the recognizer's OWN OUTPUT for the preceding words,
never the ground truth -- using the neighbours' transcriptions would leak the
test labels.

Tokens are case-sensitive (word accuracy is case-sensitive) and no sentence
boundary symbols are used: IAM lines are physical lines, not sentences, so a
line start carries no linguistic boundary.  Context is reset to "unknown"
(unigram scoring) at a line start and wherever the preceding word index is
missing from the split (words whose segmentation IAM flags as not ``ok`` are
not part of the Aachen word lists).

Interpolated KN, orders 3/2/1 (Chen & Goodman 1999, eq. for KN-interp):

    P(w3|w1 w2) = max(c(w1 w2 w3) - D, 0) / c(w1 w2 .)
                  + D * N1+(w1 w2 .) / c(w1 w2 .) * P(w3|w2)
    P(w3|w2)    = max(N1+(. w2 w3) - D, 0) / N1+(. w2 .)
                  + D * N1+(w2 .) / N1+(. w2 .) * P(w3)
    P(w3)       = (N1+(. w3) + beta) / (N1+(. .) + beta * V)

with D = 0.75 and a small additive floor (beta = 1 over the union of the LM
vocabulary and the correction lexicon) so that lexicon words never seen in the
corpus keep a positive probability.  If a context was never observed the model
falls to the next lower order.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class KNTrigram:
    def __init__(self, sentences: Iterable[Sequence[str]], discount: float = 0.75,
                 extra_vocab: Optional[Iterable[str]] = None, beta: float = 1.0):
        self.D = float(discount)
        self.beta = float(beta)
        c3: Counter = Counter()
        c2: Counter = Counter()
        c1: Counter = Counter()
        for s in sentences:
            toks = list(s)
            for t in toks:
                c1[t] += 1
            for i in range(1, len(toks)):
                c2[(toks[i - 1], toks[i])] += 1
            for i in range(2, len(toks)):
                c3[(toks[i - 2], toks[i - 1], toks[i])] += 1
        self.c3 = c3
        # trigram-context totals and "types following (w1,w2)"
        self.ctx2: Counter = Counter()          # c(w1 w2 .)
        self.follow2: Counter = Counter()       # N1+(w1 w2 .)
        cont2: Counter = Counter()              # N1+(. w2 w3)
        for (w1, w2, w3), c in c3.items():
            self.ctx2[(w1, w2)] += c
            self.follow2[(w1, w2)] += 1
            cont2[(w2, w3)] += 1
        self.cont2 = cont2
        self.contmid: Counter = Counter()       # N1+(. w2 .)
        self.follow1: Counter = Counter()       # N1+(w2 .)
        for (w2, w3), c in cont2.items():
            self.contmid[w2] += c
            self.follow1[w2] += 1
        self.cont1: Counter = Counter()         # N1+(. w3)
        for (w2, w3) in c2:
            self.cont1[w3] += 1
        self.n_bigram_types = len(c2)           # N1+(. .)
        vocab = set(c1)
        if extra_vocab is not None:
            vocab |= set(extra_vocab)
        self.V = len(vocab)
        self.n_tokens = sum(c1.values())
        self.n_types = len(c1)
        self._uni_cache: Dict[str, float] = {}

    # ---- probabilities --------------------------------------------------
    def p_uni(self, w: str) -> float:
        p = self._uni_cache.get(w)
        if p is None:
            p = (self.cont1.get(w, 0) + self.beta) / (self.n_bigram_types + self.beta * self.V)
            self._uni_cache[w] = p
        return p

    def p_bi(self, w3: str, w2: str) -> float:
        tot = self.contmid.get(w2, 0)
        if tot == 0:
            return self.p_uni(w3)
        num = max(self.cont2.get((w2, w3), 0) - self.D, 0.0)
        lam = self.D * self.follow1.get(w2, 0) / tot
        return num / tot + lam * self.p_uni(w3)

    def p_tri(self, w3: str, w1: Optional[str], w2: Optional[str]) -> float:
        if w2 is None:
            return self.p_uni(w3)
        if w1 is None:
            return self.p_bi(w3, w2)
        tot = self.ctx2.get((w1, w2), 0)
        if tot == 0:
            return self.p_bi(w3, w2)
        num = max(self.c3.get((w1, w2, w3), 0) - self.D, 0.0)
        lam = self.D * self.follow2.get((w1, w2), 0) / tot
        return num / tot + lam * self.p_bi(w3, w2)

    def logp(self, w3: str, w1: Optional[str] = None, w2: Optional[str] = None) -> float:
        return math.log(self.p_tri(w3, w1, w2))

    def describe(self) -> str:
        return (f"KN trigram: {self.n_tokens:,} tokens, {self.n_types:,} types, "
                f"{len(self.c3):,} trigrams, {self.n_bigram_types:,} bigrams, "
                f"V={self.V:,}, D={self.D}")


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------
def iam_line_segments(words_file: str) -> List[List[str]]:
    """Contiguous runs of consecutive words on the same IAM line, in order.

    A run is broken wherever a word index is missing (the Aachen lists keep
    only words whose segmentation is flagged ``ok``), so no n-gram is ever
    counted across a gap.  Transcriptions with internal spaces (e.g. "B B C")
    are joined as the recognizer sees them.
    """
    per_line: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    with open(words_file, encoding="utf-8") as fh:
        for l in fh:
            l = l.strip()
            if not l or l.startswith("#"):
                continue
            parts = l.split()
            if len(parts) < 9:
                continue
            wid = parts[0]
            line_id = "-".join(wid.split("-")[:3])
            idx = int(wid.split("-")[3])
            word = "".join(parts[8:])
            per_line[line_id].append((idx, word))
    segments: List[List[str]] = []
    for line_id, items in per_line.items():
        items.sort()
        run: List[str] = []
        prev = None
        for idx, w in items:
            if prev is not None and idx != prev + 1:
                if run:
                    segments.append(run)
                run = []
            run.append(w)
            prev = idx
        if run:
            segments.append(run)
    return segments


def brown_sentences() -> List[List[str]]:
    from nltk.corpus import brown
    return [list(s) for s in brown.sents()]


# ---------------------------------------------------------------------------
# Line-context corrector
# ---------------------------------------------------------------------------
class ContextCorrector:
    """Same acceptance rule and candidate set as TrigramLanguageModel.correct_word
    (in-lexicon hypotheses are accepted unchanged; otherwise lexicon entries
    within Levenshtein distance 1 for |w|<=4, else 2), but the candidates are
    ranked by  log P_KN(candidate | previous two output words) - alpha * dist.
    Ties are broken exactly as in correct_word: shorter candidate length first,
    then lexicographic order within a length bucket."""

    def __init__(self, lexicon_lm, kn: KNTrigram):
        self.lex = lexicon_lm          # TrigramLanguageModel (for vocabulary + buckets)
        self.kn = kn
        self._cands: Dict[str, List[Tuple[str, int]]] = {}
        self._ensure_buckets()

    def _ensure_buckets(self):
        lex = self.lex
        by_len = lex.__dict__.get("_vocab_by_len")
        if by_len is None or lex.__dict__.get("_vocab_by_len_n") != len(lex.vocabulary):
            by_len = {}
            for v in lex.vocabulary:
                by_len.setdefault(len(v), []).append(v)
            for k in by_len:
                by_len[k].sort()
            lex._vocab_by_len = by_len
            lex._vocab_by_len_n = len(lex.vocabulary)
        self.by_len = by_len

    def in_lexicon(self, w: str) -> bool:
        return w in self.lex.vocabulary or w.lower() in self.lex.vocabulary_lower

    def candidates(self, w: str) -> List[Tuple[str, int]]:
        """(candidate, distance) within the edit bound, in tie-break order.
        Independent of context and alpha, so memoised per hypothesis."""
        c = self._cands.get(w)
        if c is not None:
            return c
        import numpy as np
        max_dist = 1 if len(w) <= 4 else 2
        out: List[Tuple[str, int]] = []
        for L in range(len(w) - max_dist, len(w) + max_dist + 1):
            bucket = self.by_len.get(L, ())
            if not bucket:
                continue
            dists = self.lex._bucket_distances(w, L, bucket, max_dist)
            for i in np.nonzero(dists <= max_dist)[0]:
                out.append((bucket[int(i)], int(dists[int(i)])))
        self._cands[w] = out
        return out

    def correct_one(self, w: str, h1: Optional[str], h2: Optional[str], alpha: float,
                    use_context: bool) -> str:
        if self.in_lexicon(w):
            return w
        best, best_score = w, None
        for cand, dist in self.candidates(w):
            lp = self.kn.logp(cand, h1, h2) if use_context else self.kn.logp(cand)
            score = lp - alpha * dist
            if best_score is None or score > best_score:     # strict: keeps first on ties
                best, best_score = cand, score
        return best

    def correct_lines(self, rows: Sequence[Tuple[str, int, str]], alpha: float,
                      use_context: bool,
                      context_source: Optional[Sequence[str]] = None) -> List[str]:
        """rows: (line_id, word_idx, hypothesis) in any order; returns outputs
        aligned with rows.  Words of a line are processed in reading order and
        each output becomes context for the next word.  If `context_source`
        is given (e.g. ground truth, for an ORACLE diagnostic), it replaces the
        model's own outputs as context; never use that for a reported result."""
        order = sorted(range(len(rows)), key=lambda i: (rows[i][0], rows[i][1]))
        out: List[Optional[str]] = [None] * len(rows)
        prev_line, prev_idx = None, None
        h1: Optional[str] = None      # word two back
        h2: Optional[str] = None      # word one back
        for i in order:
            line_id, idx, hyp = rows[i]
            if line_id != prev_line or prev_idx is None or idx != prev_idx + 1:
                h1, h2 = None, None            # line start or gap: no context
            o = self.correct_one(hyp, h1, h2, alpha, use_context)
            out[i] = o
            ctx_word = context_source[i] if context_source is not None else o
            h1, h2 = h2, ctx_word
            prev_line, prev_idx = line_id, idx
        return out  # type: ignore[return-value]

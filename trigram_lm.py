#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trigram Language Model for CRNN Word Correction
Simple n-gram model for post-processing CRNN predictions
"""

import os
import math
import numpy as np


class TrigramLanguageModel:
    """
    Simple trigram language model for word-level correction.
    Builds unigram, bigram, and trigram probabilities from vocabulary.

    V3 EXTENSION (Aachen-tuned, validated +4.80pp on V2 test, McNemar p=10^-33):
      - Vocabulary = IAM training words UNION NLTK English words (~238K total)
      - This prevents over-aggressive corrections on valid English words that
        happen to be missing from the small IAM training vocabulary.
      - Without NLTK extension: trigram hurts 272 valid words on V2 Aachen test
      - With NLTK extension: only 112 hurt cases (3.29x helped/hurt ratio)
    """
    def __init__(self, words_file, use_nltk_extension=True):
        """
        Initialize trigram model from IAM words.txt
        Args:
            words_file: Path to words.txt file
            use_nltk_extension: If True, augment vocabulary with NLTK English words.
                Set False for V2-style smaller-vocab behaviour (mainly for ablation).
        """
        self.unigrams = {}
        self.bigrams = {}
        self.trigrams = {}
        self.vocabulary = set()
        self.vocabulary_lower = set()  # for case-insensitive valid-word check
        self.total_words = 0
        self.use_nltk_extension = use_nltk_extension

        # Load vocabulary and build n-grams
        self._build_model(words_file)

        # V3: Augment with NLTK English wordlist
        if use_nltk_extension:
            self._extend_with_nltk()
        
    def _build_model(self, words_file):
        """Build n-gram model from words.txt or CSV"""
        print("[TrigramLM] Building language model...")
        
        words = []
        
        if words_file.endswith('.csv'):
            import csv
            try:
                with open(words_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None) # Skip header
                    for row in reader:
                        if len(row) >= 2:
                            sentence = row[1]
                            # Remove punctuation roughly to get words
                            for char in '.,";:!?()':
                                sentence = sentence.replace(char, ' ')
                            sentence_words = sentence.split()
                            words.extend(sentence_words)
                            for w in sentence_words:
                                self.vocabulary.add(w)
            except Exception as e:
                print(f"[TrigramLM] Error reading CSV: {e}")
                
        else:
            # Original words.txt parsing
            with open(words_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) < 9:
                        continue
                    
                    # Extract word (last column)
                    word = parts[-1]
                    words.append(word)
                    self.vocabulary.add(word)
        
        self.total_words = len(words)
        
        # Build unigrams
        for word in words:
            self.unigrams[word] = self.unigrams.get(word, 0) + 1
        
        # Build bigrams
        for i in range(len(words) - 1):
            bigram = (words[i], words[i+1])
            self.bigrams[bigram] = self.bigrams.get(bigram, 0) + 1
        
        # Build trigrams
        for i in range(len(words) - 2):
            trigram = (words[i], words[i+1], words[i+2])
            self.trigrams[trigram] = self.trigrams.get(trigram, 0) + 1
        
        # Initialize lowercase mirror for valid-word check
        self.vocabulary_lower = {w.lower() for w in self.vocabulary}

        print(f"[TrigramLM] Loaded {len(self.vocabulary)} unique IAM words")
        print(f"[TrigramLM] Total words: {self.total_words}")
        print(f"[TrigramLM] Bigrams: {len(self.bigrams)}, Trigrams: {len(self.trigrams)}")

    def _extend_with_nltk(self):
        """V3: Augment vocabulary with NLTK English wordlist (235K words).
        Only affects the OOV-check set; does NOT change n-gram statistics
        (those still come from IAM training corpus).
        """
        try:
            import nltk
            try:
                from nltk.corpus import words as nltk_words
                _ = nltk_words.words()
            except LookupError:
                print("[TrigramLM] Downloading NLTK words corpus...")
                nltk.download('words', quiet=True)
                from nltk.corpus import words as nltk_words

            nltk_vocab = set(nltk_words.words())
            before = len(self.vocabulary)
            self.vocabulary |= nltk_vocab
            self.vocabulary_lower |= {w.lower() for w in nltk_vocab}
            print(f"[TrigramLM] Extended vocabulary with NLTK: "
                  f"{before:,} -> {len(self.vocabulary):,} unique words "
                  f"(+{len(self.vocabulary)-before:,} from NLTK)")
        except ImportError:
            print("[TrigramLM] WARNING: NLTK not installed - skipping vocabulary extension")
        except Exception as e:
            print(f"[TrigramLM] WARNING: NLTK extension failed ({e}) - using IAM vocab only")

    def score_word(self, word, prev_words=None):
        """
        Score a word using n-gram probability with smoothing and backoff.
        Uses trigram if 2 previous words available, bigram if 1, unigram otherwise.
        Returns log probability
        """
        if prev_words and len(prev_words) >= 2:
            # Trigram: P(word | prev2, prev1)
            trigram_key = (prev_words[-2], prev_words[-1], word)
            if trigram_key in self.trigrams:
                bigram_count = self.bigrams.get((prev_words[-2], prev_words[-1]), 1)
                prob = (self.trigrams[trigram_key] + 1) / (bigram_count + len(self.vocabulary))
                return math.log(prob)
            # Backoff to bigram
            prev_words = [prev_words[-1]]

        if prev_words and len(prev_words) >= 1:
            # Bigram: P(word | prev1)
            bigram_key = (prev_words[-1], word)
            if bigram_key in self.bigrams:
                prob = (self.bigrams[bigram_key] + 1) / (self.unigrams.get(prev_words[-1], 1) + len(self.vocabulary))
                return math.log(prob)

        # Unigram fallback: P(word)
        if word in self.unigrams:
            prob = (self.unigrams[word] + 1) / (self.total_words + len(self.vocabulary))
        else:
            prob = 1 / (self.total_words + len(self.vocabulary))

        return math.log(prob)
    
    def score_sequence(self, words):
        """
        Score a sequence of words using trigram model
        Returns log probability
        """
        if not words:
            return 0.0
        
        score = 0.0
        
        # Unigram for first word
        score += self.score_word(words[0])
        
        # Bigram for second word if exists
        if len(words) > 1:
            bigram = (words[0], words[1])
            if bigram in self.bigrams:
                prob = (self.bigrams[bigram] + 1) / (self.unigrams[words[0]] + len(self.vocabulary))
            else:
                prob = 1 / (self.unigrams.get(words[0], 1) + len(self.vocabulary))
            score += math.log(prob)
        
        # Trigram for remaining words
        for i in range(2, len(words)):
            trigram = (words[i-2], words[i-1], words[i])
            if trigram in self.trigrams:
                bigram_count = self.bigrams.get((words[i-2], words[i-1]), 1)
                prob = (self.trigrams[trigram] + 1) / (bigram_count + len(self.vocabulary))
            else:
                # Backoff to bigram
                bigram = (words[i-1], words[i])
                if bigram in self.bigrams:
                    prob = (self.bigrams[bigram] + 1) / (self.unigrams.get(words[i-1], 1) + len(self.vocabulary))
                else:
                    # Backoff to unigram
                    prob = (self.unigrams.get(words[i], 0) + 1) / (self.total_words + len(self.vocabulary))
            score += math.log(prob)
        
        return score
    
    def correct_word(self, word, prev_words=None, max_candidates=5):
        """
        Find closest matching word in vocabulary using edit distance
        and n-gram context (trigram/bigram/unigram with backoff).

        V3 STRATEGY (Aachen+NLTK, validated +4.80pp on V2 test, McNemar p=10^-33):
          - Vocabulary = IAM training + NLTK English wordlist (~238K total)
          - Case-insensitive validity check (handles capitalised proper nouns)
          - Tight edit-distance bounds: d_max = 1 (|w|<=4), 2 (5<=|w|<=8), 2 (|w|>8)
          - Edit penalty: alpha = 5.0
          - Hurt cases reduced from 272 (V2 IAM-only) to 112 (V3 IAM+NLTK)

        Args:
            word: The word to correct
            prev_words: List of previous words for trigram/bigram context
        Returns the best matching word as a string
        """
        # Memo: the same string arrives thousands of times during validation;
        # the result is deterministic (when prev_words is None), so compute once.
        _cache = self.__dict__.setdefault("_correct_cache", {})
        if prev_words is None and word in _cache:
            return _cache[word]

        # V3: Recognize valid English words (case-insensitive)
        if word in self.vocabulary or word.lower() in self.vocabulary_lower:
            if prev_words is None:
                _cache[word] = word
            return word

        # V2: Tighter dynamic edit distance threshold
        if len(word) <= 4:
            max_dist = 1
        elif len(word) <= 8:
            max_dist = 2
        else:
            max_dist = 2

        # Index the vocabulary by length (built once, after the NLTK extension).
        # Entries with |len(word)-len(v)| > max_dist were skipped anyway, so
        # visiting only the admissible length buckets yields the SAME candidate
        # set as scanning all 239K words; scoring and selection are unchanged.
        by_len = self.__dict__.get("_vocab_by_len")
        if by_len is None or self.__dict__.get("_vocab_by_len_n") != len(self.vocabulary):
            by_len = {}
            for v in self.vocabulary:
                by_len.setdefault(len(v), []).append(v)
            self._vocab_by_len = by_len
            self._vocab_by_len_n = len(self.vocabulary)

        candidates = []
        for L in range(len(word) - max_dist, len(word) + max_dist + 1):
            bucket = by_len.get(L, ())
            if not bucket:
                continue
            # Exact Levenshtein for the whole bucket at once (numpy DP).
            # Same distances as self._edit_distance, same candidate order.
            dists = self._bucket_distances(word, L, bucket, max_dist)
            for idx in np.nonzero(dists <= max_dist)[0]:
                vocab_word = bucket[idx]
                dist = int(dists[idx])
                # V2: Higher edit penalty (alpha=5.0) -> prefer fewer edits
                score = self.score_word(vocab_word, prev_words=prev_words) - dist * 5.0
                candidates.append((vocab_word, score))

        # Sort by score and return best candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_word = candidates[0][0]
        else:
            best_word = word  # Return original if no candidates found
        if prev_words is None:
            _cache[word] = best_word
        return best_word
    
    def _bucket_distances(self, word, L, bucket, max_dist=None):
        """Levenshtein distance from `word` to every entry of a same-length
        bucket, computed with a vectorised DP (rows = characters of `word`,
        columns = positions in the bucket words, batched over the bucket).
        Bit-for-bit the same integers as _edit_distance for every candidate
        whose distance is <= max_dist. Candidates that can no longer reach
        max_dist (their DP row minimum already exceeds it -- row minima never
        decrease) are pruned early and reported as max_dist+1."""
        import numpy as np
        codes = self.__dict__.setdefault("_bucket_codes", {})
        entry = codes.get(L)
        if entry is None or entry[0] is not bucket:
            M = np.zeros((len(bucket), L), dtype=np.int32)
            for i, v in enumerate(bucket):
                M[i, :] = [ord(c) for c in v]
            codes[L] = (bucket, M)
        else:
            M = entry[1]
        N = M.shape[0]
        q = np.fromiter((ord(c) for c in word), dtype=np.int32, count=len(word))
        cap = None if max_dist is None else int(max_dist)
        out = np.full(N, (cap + 1) if cap is not None else 0, dtype=np.int32)
        alive = np.arange(N)                       # indices still in play
        Mv = M
        prev = np.tile(np.arange(L + 1, dtype=np.int32), (N, 1))      # row 0
        for i in range(1, len(word) + 1):
            cur = np.empty_like(prev)
            cur[:, 0] = i
            sub = prev[:, :-1] + (Mv != q[i - 1])                    # substitution
            dele = prev[:, 1:] + 1                                    # deletion
            best = np.minimum(sub, dele)
            for j in range(1, L + 1):                                 # insertion
                cur[:, j] = np.minimum(best[:, j - 1], cur[:, j - 1] + 1)
            prev = cur
            if cap is not None:
                keep = prev.min(axis=1) <= cap
                if not keep.all():
                    alive = alive[keep]
                    prev = prev[keep]
                    Mv = Mv[keep]
                    if alive.size == 0:
                        return out
        out[alive] = prev[:, L]
        return out

    def _edit_distance(self, s1, s2):
        """Calculate Levenshtein edit distance"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

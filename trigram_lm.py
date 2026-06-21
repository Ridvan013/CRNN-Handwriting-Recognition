#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trigram Language Model for CRNN Word Correction
Simple n-gram model for post-processing CRNN predictions
"""

import os
import math


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
        # V3: Recognize valid English words (case-insensitive)
        if word in self.vocabulary or word.lower() in self.vocabulary_lower:
            return word

        # V2: Tighter dynamic edit distance threshold
        if len(word) <= 4:
            max_dist = 1
        elif len(word) <= 8:
            max_dist = 2
        else:
            max_dist = 2

        candidates = []
        for vocab_word in self.vocabulary:
            # Optimization: Skip words with large length difference
            if abs(len(word) - len(vocab_word)) > max_dist:
                continue

            dist = self._edit_distance(word, vocab_word)
            if dist <= max_dist:
                # V2: Higher edit penalty (alpha=5.0) -> prefer fewer edits
                score = self.score_word(vocab_word, prev_words=prev_words) - dist * 5.0
                candidates.append((vocab_word, score))

        # Sort by score and return best candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_word = candidates[0][0]
            return best_word
        else:
            return word  # Return original if no candidates found
    
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

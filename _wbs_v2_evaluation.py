"""
Apply Word Beam Search decoder to V2 model on Aachen test set.
Compares WBS vs greedy/beam/trigram outputs.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

# Suppress OpenCV warnings
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

from wbs_pure_python import WordBeamSearchDecoder
import nltk
try:
    from nltk.corpus import words as nltk_words
    _ = nltk_words.words()
except LookupError:
    nltk.download('words', quiet=True)
    from nltk.corpus import words as nltk_words

CHAR_LIST = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# ============================================================
# V2 Model Definition (must match greedy_aachen_v2.py)
# ============================================================
class CRNNModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = len(CHAR_LIST) + 1
        self.hidden = 384
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d((2,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d((2,1)),
            nn.Conv2d(512, 512, 2, padding=0), nn.ReLU(inplace=True)
        )
        self.rnn = nn.LSTM(input_size=512, hidden_size=self.hidden, num_layers=3,
                          bidirectional=True, batch_first=False, dropout=0.2)
        self.classifier = nn.Linear(2 * self.hidden, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        rnn_in = conv.squeeze(2).permute(2, 0, 1)
        lstm_out, _ = self.rnn(rnn_in)
        logits = self.classifier(lstm_out)
        return F.log_softmax(logits, dim=2)


def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Convert to tensor
    img_t = torch.from_numpy(img).float() / 255.0
    img_t = 1.0 - img_t  # invert
    img_t = (img_t - 0.5) / 0.5
    img_t = img_t.unsqueeze(0).unsqueeze(0)
    img_t = F.interpolate(img_t, size=(32, 128), mode='bilinear', align_corners=False)
    return img_t.squeeze(0)  # [1, 32, 128]


def load_aachen_test_samples():
    """Load test set samples: (word_id, transcription, image_path)"""
    samples = []
    aachen_dir = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\aachen_splits"
    img_root = r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\HTR_Using_CRNN\IAM\processed\archive\iam_words\words"
    with open(os.path.join(aachen_dir, "test_words.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            word_id = parts[0]
            transcription = parts[-1]
            a, b = word_id.split("-")[:2]
            img_path = os.path.join(img_root, a, f"{a}-{b}", f"{word_id}.png")
            if os.path.exists(img_path):
                samples.append((word_id, transcription, img_path))
    return samples


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print("Loading V2 model...")
    model = CRNNModelV2().to(device)
    ckpt = torch.load(r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\Model_aachen_v2\best_model_wa.pth", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # Load vocabulary
    print("Loading vocabulary (IAM Aachen train + NLTK English)...")
    iam_vocab = set()
    iam_words_for_lm = []
    with open(r"c:\Users\RIDVAN\Desktop\CRNN\CRNN_1\aachen_splits\train_words.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 9:
                w = parts[-1]
                iam_vocab.add(w)
                iam_words_for_lm.append(w)

    nltk_vocab = set(nltk_words.words())
    full_vocab = iam_vocab | nltk_vocab
    # Add lowercase versions for case-insensitive WBS coverage
    full_vocab_with_case = full_vocab | {w.lower() for w in full_vocab} | {w.capitalize() for w in full_vocab}
    print(f"  IAM:    {len(iam_vocab):,}")
    print(f"  NLTK:   {len(nltk_vocab):,}")
    print(f"  Merged: {len(full_vocab_with_case):,} (incl. case variants)")

    # Filter to chars used by CRNN
    char_set = set(CHAR_LIST)
    filtered_vocab = [w for w in full_vocab_with_case if w and all(c in char_set for c in w)]
    print(f"  After char-filter: {len(filtered_vocab):,}")

    # Build WBS decoder
    print("Building Word Beam Search decoder (NGrams mode)...")
    wbs = WordBeamSearchDecoder(
        char_list=CHAR_LIST,
        vocabulary=filtered_vocab,
        beam_width=25,
        mode="NGrams",
        lm_weight=0.7,
        lm_words=iam_words_for_lm,  # char bigram from IAM only (smaller, focused)
    )

    # Load test samples
    samples = load_aachen_test_samples()
    print(f"Loaded {len(samples)} test samples")

    # Run inference
    BATCH = 64
    correct_wbs = 0
    total = 0
    sample_predictions = []

    print("Running inference + WBS decoding...")
    with torch.no_grad():
        for i in range(0, len(samples), BATCH):
            batch = samples[i:i + BATCH]
            imgs = []
            for word_id, true, path in batch:
                t = preprocess_image(path)
                if t is None:
                    continue
                imgs.append(t)
            if not imgs:
                continue
            batch_tensor = torch.stack(imgs).to(device)
            log_probs = model(batch_tensor)  # [T, B, C]

            input_lens = [log_probs.size(0)] * log_probs.size(1)
            decoded = wbs.decode_batch(log_probs, input_lens)

            for k, (word_id, true, _) in enumerate(batch):
                if k >= len(decoded):
                    continue
                pred = decoded[k]
                total += 1
                if pred == true:
                    correct_wbs += 1
                if len(sample_predictions) < 20:
                    sample_predictions.append((true, pred, pred == true))

            if (i + BATCH) % 500 == 0 or (i + BATCH) >= len(samples):
                print(f"  Processed {min(i + BATCH, len(samples))}/{len(samples)}, "
                      f"WA so far: {correct_wbs/total*100:.2f}%")

    wbs_wa = correct_wbs / total

    print(f"\n{'='*72}")
    print(f"  WORD BEAM SEARCH (NGrams) RESULTS - V2 MODEL ON AACHEN TEST")
    print(f"{'='*72}")
    print(f"  Samples evaluated      : {total}")
    print(f"  WBS Word Accuracy      : {wbs_wa*100:.2f}%   ({correct_wbs}/{total})")
    print()
    print(f"  Reference (from prior runs):")
    print(f"    Raw Greedy           :  71.73%   (3829/5338)")
    print(f"    Greedy + Smart Trigr :  72.56%   (3873/5338)")
    print(f"    Greedy + V3 Extended :  76.53%   (4085/5338)")
    print(f"    Beam + V3 Extended   :  76.85%   (4102/5338)")
    print()
    print(f"  Sample predictions (first 20):")
    print(f"    {'TRUE':<20s} {'WBS PRED':<20s} {'OK?'}")
    for true, pred, ok in sample_predictions:
        marker = '✓' if ok else '✗'
        print(f"    {true:<20s} {pred:<20s} {marker}")


if __name__ == "__main__":
    main()

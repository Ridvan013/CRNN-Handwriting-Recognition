"""
Kaggle path configuration for H2 pipeline.

Dataset: brht25/paper-traning-data  (private, IAM Handwriting Database)
Kaggle'da notebook'a eklerken: Add Data → Your Datasets → paper_traning_data
"""

import os
from pathlib import Path

IN_KAGGLE = os.path.exists("/kaggle/input")

# ── IAM Dataset slug ──────────────────────────────────────────────────────────
IAM_DATASET_SLUG = "paper-traning-data"   # brht25/paper-traning-data

# ── Path constants ────────────────────────────────────────────────────────────
if IN_KAGGLE:
    _INPUT   = Path("/kaggle/input")
    _WORKING = Path("/kaggle/working")

    IAM_INPUT_DIR = _INPUT / IAM_DATASET_SLUG

    # Dataset içindeki klasör yapısını otomatik bul
    # archive/ altında veya doğrudan kök'te olabilir
    _candidates_txt = [
        IAM_INPUT_DIR / "iam_words" / "words.txt",
        IAM_INPUT_DIR / "archive" / "iam_words" / "words.txt",
        IAM_INPUT_DIR / "words.txt",
        IAM_INPUT_DIR / "archive" / "words.txt",
    ]
    _candidates_dir = [
        IAM_INPUT_DIR / "iam_words" / "words",
        IAM_INPUT_DIR / "archive" / "iam_words" / "words",
        IAM_INPUT_DIR / "words",
        IAM_INPUT_DIR / "archive" / "words",
    ]

    _found_txt = next((str(p) for p in _candidates_txt if p.exists()), None)
    _found_dir = next((str(p) for p in _candidates_dir if p.exists()), None)

    IAM_WORDS_TXT = _found_txt or str(_candidates_txt[0])
    IAM_WORDS_DIR = _found_dir or str(_candidates_dir[0])

    SYNTH_DIR  = str(_WORKING / "synthetic_data")
    FONT_DIR   = str(_WORKING / "fonts")
    CKPT_DIR   = str(_WORKING / "checkpoints")
    RESULTS    = str(_WORKING / "results")
    LOGS       = str(_WORKING / "logs")
else:
    IAM_WORDS_TXT = ""
    IAM_WORDS_DIR = ""
    SYNTH_DIR  = "synthetic_data"
    FONT_DIR   = "cloud/fonts"
    CKPT_DIR   = "checkpoints"
    RESULTS    = "results"
    LOGS       = "logs"


def print_paths():
    print(f"IN_KAGGLE   : {IN_KAGGLE}")
    print(f"IAM words   : {IAM_WORDS_TXT}")
    print(f"IAM images  : {IAM_WORDS_DIR}")
    print(f"Synth data  : {SYNTH_DIR}")
    print(f"Checkpoints : {CKPT_DIR}")
    print(f"Results     : {RESULTS}")


def verify_iam_paths():
    """IAM path'lerinin var olduğunu kontrol eder, yoksa ne yapılacağını söyler."""
    if not IN_KAGGLE:
        return
    if not os.path.exists(IAM_WORDS_TXT):
        print(f"  Dataset içeriği:")
        for p in sorted(Path("/kaggle/input").rglob("*.txt")):
            print(f"    {p}")
        raise FileNotFoundError(
            f"IAM words.txt bulunamadı.\n"
            f"  Yukarıdaki listeden doğru path'i bul,\n"
            f"  kaggle_paths.py'deki _candidates_txt listesine ekle."
        )
    if not os.path.exists(IAM_WORDS_DIR):
        raise FileNotFoundError(
            f"IAM words/ dizini bulunamadı: {IAM_WORDS_DIR}"
        )
    print(f"  IAM paths OK")

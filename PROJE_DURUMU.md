# El Yazısı Tanıma Projesi — Mevcut Durum ve Yapılacaklar

## 🎯 Projenin Amacı

IAM Handwriting Database üzerinde **resource-efficient** (sınırlı kaynakla çalışan)
HTR (Handwritten Text Recognition / El Yazısı Tanıma) sistemi geliştiriyoruz.

**Constraint**: Sadece CRNN mimarisi (Convolutional Recurrent Neural Network).
TrOCR/Transformer gibi büyük modeller değil — pre-training gerektirmeyen, tek
GPU'da eğitilebilen sistem.

**Hedef**: Aachen writer-disjoint split'te **%85+ Word Accuracy** (WA).

## 📊 Şu Anki Sonuçlar

### Aachen Test Set (5,338 sample, writer-disjoint)

| Model | Decoder | Trigram | Test WA |
|---|---|---|---|
| V1 (8.75M params) | Greedy | V1 (loose, IAM-only) | 70.29% |
| V1 (8.75M params) | Greedy | V2 (smart, IAM-only) | 72.05% |
| **V2 (15.46M params)** | Greedy | V3 (NLTK extended, 238K vocab) | **76.53%** |
| **V2 (15.46M params)** | Beam k=10 | V3 | **76.85%** |

### Şu An Eğitilmekte: V3 Model (28.73M params)

- 4-layer BiLSTM hidden=512 (V2: 3-layer h=384)
- Cosine LR schedule + 5-epoch warmup
- AdamW optimizer + weight decay
- V3 trigram (NLTK + IAM) in-loop validation
- **Hedef**: %78-82 (V2'nin +2-5pp üzerinde)

## 🔬 Kullanılan Yöntemler

### 1. Aachen Writer-Disjoint Split
IAM'in standart "every-10th word" custom split'i writer-leakage içeriyor
(aynı yazarın kelimeleri hem train hem val'da olabiliyor).

**Aachen split** (Bluche/RWTH, OpenSLR-56) ile değiştirdik:
- 747 form → Train (31,615 word, +unassigned IAM forms)
- 116 form → Validation (1,646 word, farklı yazarlar)
- 336 form → Test (5,338 word, üçüncü farklı yazar grubu)

### 2. V3 Extended Trigram Language Model
**Bulgu**: V1 (loose, d_max=2/3/4) trigram Aachen'de **ZARAR** veriyor (-0.36pp,
p=0.50). Sebep: model'in doğru tahmin ettiği valid İngilizce kelimeleri (writer,
comment, telephone) IAM-only sözlükte olmadığı için "OOV" sayıp yanlış kelimelere
düzeltiyor.

**Çözüm V3**:
- Vocabulary: IAM Aachen train (5,939) + NLTK English wordlist (235,892) = **238K kelime**
- Tight edit distance bounds: d_max = 1 (|w|≤4), 2 (5≤|w|≤8), 2 (|w|>8)
- Edit penalty alpha = 5.0
- Case-insensitive valid-word check

**Sonuç**: V2 modelde +4.80pp greedy / +5.11pp beam, McNemar p=1.27×10⁻³⁵.

### 3. Word Beam Search Decoder (WBS)
Scheidl-Fiel-Sablatnig ICFHR 2018 implementasyonu — `word_beam_search.py`.
Lexicon-aware CTC decoder, karakter bigram LM ile.

**Bulgu**: V2 modelde WBS +0.03pp marjinal (76.56% vs trigram'ın 76.53%'ü).
WBS ve V3 trigram aynı işi yapıyor (sözlük-bazlı düzeltme). V3 model'de tekrar
test edilecek.

### 4. İstatistiksel Anlamlılık
Her ablation için McNemar test (paired) + Wilson 95% confidence interval.
- V2 raw vs V3 trigram: p = 7.04 × 10⁻³³ (extreme significant)
- V2 raw vs Beam+V3: p = 1.27 × 10⁻³⁵
- Reviewer-safe statistical claims.

## 📂 Repository Yapısı

```
CRNN_1/
├── greedy.py                      # V0 - Orijinal custom-split (referans)
├── greedy_aachen.py               # V1 - Aachen split (8.75M params)
├── greedy_aachen_v2.py            # V2 - 3-layer BiLSTM (15.46M params)
├── greedy_aachen_v3.py            # V3 - 4-layer BiLSTM (28.73M params) + cosine LR
├── trigram_lm.py                  # V3 trigram (IAM + NLTK)
├── word_beam_search.py            # WBS NGrams decoder
├── pipeline_v2.py                 # CRAFT + CRNN production pipeline
├── build_aachen_word_splits.py    # Aachen form -> word mapping
├── aachen_splits/
│   ├── splits/
│   │   ├── train.uttlist         # 747 form IDs
│   │   ├── validation.uttlist    # 116 form IDs
│   │   └── test.uttlist          # 336 form IDs
│   └── {train,validation,test}_words.txt  # word-level lists
├── Model_aachen/                  # V1 trained model (~33MB each .pth)
├── Model_aachen_v2/               # V2 trained model + analysis
├── Model_aachen_v3/               # V3 - şu an eğitiliyor (henüz yok)
├── _beam_search_test.py           # WBS sanity tests
├── _mcnemar_extended.py           # 5+ paired McNemar tests
├── _trigram_strategies.py         # 6 trigram strategy comparison
├── _trigram_with_nltk.py          # V3 NLTK extension test
├── _analyze_remaining_hurt.py     # Hurt case breakdown
├── _wbs_v2_evaluation.py          # WBS post-hoc evaluation
└── PROJE_DURUMU.md                # bu dosya
```

## 🚀 Reproduction — Sıfırdan Başlama

### Gereksinimler
- Python 3.10+
- PyTorch 2.0+ with CUDA
- IAM Handwriting Database (words/ klasörü ile, words.txt dahil)
- NLTK words corpus

### Kurulum
```bash
git clone https://github.com/Ridvan013/CRNN-Handwriting-Recognition.git
cd CRNN-Handwriting-Recognition
git checkout feature/aachen-v3-extended-trigram

pip install torch torchvision opencv-python numpy pandas matplotlib nltk scikit-learn
python -c "import nltk; nltk.download('words')"

# IAM dataset'i şu yola yerleştir:
# CRNN_1/HTR_Using_CRNN/IAM/processed/archive/iam_words/words.txt
# CRNN_1/HTR_Using_CRNN/IAM/processed/archive/iam_words/words/  (görüntüler)
```

### Aachen Split Hazırla
```bash
cd CRNN_1
python build_aachen_word_splits.py
# Çıktı: aachen_splits/{train,validation,test}_words.txt
```

### V1 Eğitim (~95 saniye/epoch, ~80 dakika toplam)
```bash
python greedy_aachen.py > Model_aachen_training.log 2>&1
```

### V2 Eğitim (~125 saniye/epoch, ~100 dakika toplam)
```bash
python greedy_aachen_v2.py > Model_aachen_v2_training.log 2>&1
```

### V3 Eğitim (~180 saniye/epoch, ~3 saat toplam)
```bash
python greedy_aachen_v3.py > Model_aachen_v3_training.log 2>&1
```

### Analiz (eğitim sonrası)
```bash
# McNemar testleri (5 paired comparison)
$env:CRNN_CSV = "Model_aachen_v3/test_results_analysis.csv"
python _mcnemar_extended.py

# WBS NGrams post-hoc evaluation
python _wbs_v2_evaluation.py  # V2 için, V3 versiyonu yapılacak

# Trigram strategy comparison
python _trigram_strategies.py
```

## 🔥 Şu An Senin Yardımına İhtiyacımız Olan

%85+ hedefe ulaşmak için **3 alternatif plan** var:

### Plan A — V3 Sonucunu Bekle (En Kolay)
V3 eğitimi şu an koşuyor. Bittiğinde sonuca göre devam edeceğiz.

### Plan B — Synthetic Pretraining (Yüksek Etki)
Pure CRNN için %85 zor — gerçek atılım synthetic data ile gelebilir:

1. **Synth90k / MJSynth** indirilebilir
2. Veya **TextRecognitionDataGenerator** ile el yazısı fontlarıyla 100k-500k
   synthetic word image üretilebilir
3. Stage 1: synthetic data ile pretrain (5-10 epoch)
4. Stage 2: IAM Aachen train ile fine-tune

**Beklenen**: +3-5pp
**Süre**: ~10-15 saat (data generation + 2 stage training)

### Plan C — 3-Seed Ensemble
3 farklı seed'le V3 eğit, voting ile birleştir.

**Beklenen**: +1-2pp
**Süre**: ~9 saat (3 × 3h training)

## 🎯 Senden Beklediğimiz

Eğer farklı bir GPU'ya / makineye erişimin varsa:

1. **Repo'yu clone et** ve `feature/aachen-v3-extended-trigram` branch'ine geç
2. Aşağıdakilerden birini deneyebilirsin (sırayla):

### Öncelik 1 — Synthetic Data Generation
TextRecognitionDataGenerator (https://github.com/Belval/TextRecognitionDataGenerator)
veya benzeri bir tool ile:
- **NLTK English words listesinden** kelime al
- **El yazısı simulating fontlar** kullan (Comic Sans, Casual, Inkfree, Lobster, vb.)
- Çeşitli distortion/augmentation uygula
- 100k-500k synthetic word image üret
- Sonra `greedy_aachen_v3.py` benzeri bir script yaz: **stage1: synthetic, stage2: IAM**

### Öncelik 2 — Seed Variation (Ensemble)
`greedy_aachen_v3.py`'ı 3 farklı seed'le koş, sonuçları kaydet. Sonra average voting.

### Öncelik 3 — Architecture Experiments
CRNN constraint ile farklı CNN backbone'lar dene:
- ResNet18 (ImageNet-pretrained) → büyük CNN
- VGG16-style daha derin
- ConvNeXt-tiny

## ❓ Sorular / Tartışılacaklar

- Pretraining için synthetic data domain'i ne kadar önemli? (handwriting vs scene text)
- Daha büyük model overfit'e neden olur mu? (V3 zaten %85 daha büyük)
- WBS yerine başka decoder denemeli mi? (CTC + Transformer language model?)
- Test-time augmentation (TTA) ne kadar yardım eder?

---

**Repo**: https://github.com/Ridvan013/CRNN-Handwriting-Recognition
**Branch**: `feature/aachen-v3-extended-trigram`
**Son güncelleme**: 2026-06-08

İstediğin herhangi bir şeyi sorabilirsin, kodda her şey commented ve çalışır
durumda. İyi şanslar!

# H2 Plan — Synthetic Pretraining + Ensemble (Hedef %85+)

## 📊 Mevcut Durum (Phase 0 Sonu)

| Aşama | Test WA | Wilson 95% CI | McNemar p |
|---|---|---|---|
| V1 (8.75M params) | 70.29% | [69.05, 71.50] | — |
| V2 (15.46M params) | 76.85% | [75.69, 77.96] | 1.27e-35 |
| **V3 (28.73M params)** | **78.12%** | **[76.99, 79.21]** | — |

**Hedef**: %85+ (gap: ~6.88pp)

## 🎯 H2'nin Genel Yaklaşımı

V3 model kapasitesi yeterli; eksik olan **çeşitli yazar/font öğrenme deneyimi**.
Çözüm = synthetic pretraining (yapay veriyle önce öğret, gerçek IAM ile fine-tune).

Sonra **3-seed ensemble** ile son artış. Toplam beklenen: **+5-7pp** → %83-85.

---

# 📋 Phase Detayları

## Phase 1 — Synthetic Data Generation (~3-4 saat)

### 1.1 Tool Seçimi

**Önerilen**: [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator)
- Python kütüphanesi
- Çeşitli font + distortion + background destekler
- Çıkış: PNG image + label (txt)

**Alternatif**: PIL ile custom script (daha fazla kontrol ama daha fazla iş)

### 1.2 Veri Spec

| Özellik | Değer |
|---|---|
| Kelime kaynağı | NLTK English words (235K) + IAM train (5.9K) = 240K vocab |
| Hedef sample sayısı | 300,000 (her kelime ~1-2x) |
| Image boyutu | 1×32×128 grayscale (CRNN input ile uyumlu) |
| Font sayısı | 15-20 el yazısı simulating fontu |
| Augmentations | rotation ±10°, gaussian noise, motion blur, ink intensity |

### 1.3 Önerilen Fontlar (Handwriting-Style)

İndirilecek/kullanılacak ücretsiz fontlar:
- **Caveat** (Google Fonts) - casual handwriting
- **Indie Flower** (Google Fonts)
- **Kalam** (Google Fonts) - Indian handwriting
- **Patrick Hand** (Google Fonts)
- **Shadows Into Light** (Google Fonts)
- **Architects Daughter** (Google Fonts)
- **Permanent Marker** (Google Fonts)
- **Sacramento** (Google Fonts) - script
- **Dancing Script** (Google Fonts)
- **Pacifico** (Google Fonts)
- Windows: **Comic Sans MS**, **Segoe Script**, **Lucida Handwriting**, **Ink Free**
- Casual: **Kalam**, **Itim**, **Just Another Hand**

### 1.4 Synthetic Generation Script (yazılacak)

`generate_synthetic_data.py`:
```python
# Inputs: NLTK words, font_dir, output_dir, target_count
# For each sample:
#   1. Pick random word from vocab
#   2. Pick random font from font_dir
#   3. Render with random size (font_size 20-40)
#   4. Apply random augmentations:
#      - rotation ±10°
#      - shear ±5°
#      - gaussian noise σ=0.02-0.05
#      - motion blur kernel 3-5
#      - ink variation (slight color tweak)
#   5. Pad/crop to 32×128
#   6. Save as PNG + record label
# Output: synthetic_data/words/xxx.png + synthetic_data/labels.txt
```

### 1.5 Beklenen Çıktı
```
synthetic_data/
├── labels.txt        # word_id<space>label format
├── words/
│   ├── 000000.png
│   ├── 000001.png
│   └── ... (300K image)
```

### 1.6 Doğrulama
Generation bitince:
- 20-30 rastgele sample görsel olarak incele (gerçekçi mi?)
- Toplam image sayısı doğru mu?
- Disk kullanımı: ~3-5 GB beklenir

---

## Phase 2 — Synthetic Pretrain (~4-6 saat)

### 2.1 Script: `pretrain_synthetic.py`

V3 model'i synthetic data ile **5-10 epoch** eğit.

**Hyperparameters**:
- Optimizer: AdamW(lr=1e-3, weight_decay=1e-5)
- Scheduler: Cosine warmup 2 epoch + cosine decay
- Batch size: 128 (V3 ile aynı)
- Loss: CTCLoss
- AMP: enabled
- Epochs: 8-10
- **Trigram correction yok** (pretrain için gereksiz)

### 2.2 Validation

Synthetic data'nın küçük bir bölümünü (10K sample) validation olarak ayır.
Per-epoch val WA görmek için.

**Beklenen**: Synthetic val WA %90+ olmalı (synth data daha kolay).

### 2.3 Checkpoint Kaydet

`Model_pretrain_synthetic/best_model_wa.pth`

Bu checkpoint **Phase 3**'te starting point olacak.

---

## Phase 3 — IAM Aachen Fine-tune (~3-4 saat)

### 3.1 Script: `finetune_iam.py`

V3'ün pretrained checkpoint'ini yükle, **IAM Aachen train** üzerinde fine-tune et.

**Hyperparameters**:
- Optimizer: AdamW(lr=**1e-4**, weight_decay=1e-5) — DAHA DÜŞÜK LR önemli!
- Scheduler: Cosine warmup 3 epoch + cosine decay
- Batch size: 128
- Loss: CTCLoss
- AMP: enabled
- Epochs: 40-50
- Early stopping patience: 15
- **V3 trigram in-loop validation** (mevcut greedy_aachen_v3.py gibi)

### 3.2 Important: Loading Pretrained

```python
# Load synthetic-pretrained weights
ckpt = torch.load("Model_pretrain_synthetic/best_model_wa.pth")
model.load_state_dict(ckpt)

# Optional: Freeze CNN backbone for first 5 epochs, then unfreeze
# (helps preserve synthetic visual features)
for p in model.cnn.parameters():
    p.requires_grad = False
# After epoch 5, unfreeze:
# for p in model.cnn.parameters(): p.requires_grad = True
```

### 3.3 Checkpoint Kaydet

`Model_aachen_v3_pretrained/best_model_wa.pth`

### 3.4 Beklenen Test WA

| Configuration | Beklenen |
|---|---|
| Raw Greedy | ~78-80% |
| Greedy + V3 Trigram | **~81-83%** |
| Beam + V3 Trigram | **~81-83%** |

---

## Phase 4 — 3-Seed Ensemble (~9 saat, paralelse 3 saat)

### 4.1 Strategy

V3 fine-tune'u **3 farklı random seed** ile koş:
- Seed 42
- Seed 123
- Seed 2026

Her seed → ayrı klasör (`Model_aachen_v3_seed{X}/`)

### 4.2 Script: `train_ensemble.py`

Her seed için:
1. `torch.manual_seed(seed)` + numpy + Python random
2. Aynı pretrain checkpoint'ten başla
3. Aynı IAM Aachen üzerinde fine-tune (Phase 3 ile aynı)
4. Best checkpoint kaydet

### 4.3 Inference: Voting/Averaging

`ensemble_inference.py`:
```python
# Load 3 models
models = [load(f"Model_aachen_v3_seed{s}/best_model_wa.pth") for s in [42, 123, 2026]]

# For each test image:
#   1. Get log_probs from each model
#   2. Average log_probs (softmax averaging)
#   3. Greedy/Beam decode + V3 trigram
```

### 4.4 Beklenen Ensemble Boost

Ensemble genelde **+1-2pp** verir → ~%83-85.

---

## Phase 5 — Test-Time Augmentation (TTA, ~30 dk)

### 5.1 Strategy

Test'te her image'i **5 farklı augment** ile geçir, log_probs ortalamasını al:
- Original
- Rotation -3°
- Rotation +3°
- Scale 0.95
- Scale 1.05

### 5.2 Script: `tta_inference.py`

```python
for image in test_set:
    augs = [identity, rot_minus_3, rot_plus_3, scale_095, scale_105]
    log_probs_list = [model(aug(image)) for aug in augs]
    avg_log_probs = mean(log_probs_list)
    pred = greedy_decode(avg_log_probs) + trigram_correct(pred)
```

### 5.3 Beklenen TTA Boost

**+0.5-1.5pp** → ~%84-86.

---

# 📊 Final Beklenen Sonuçlar

| Aşama | Test WA | Cumulative |
|---|---|---|
| Phase 0 (mevcut V3) | 78.12% | 78.12% |
| Phase 3 (synthetic pretrain + fine-tune) | +3-5pp | **81-83%** |
| Phase 4 (3-seed ensemble) | +1-2pp | **82-84%** |
| Phase 5 (TTA) | +0.5-1pp | **83-85%** |

**Hedef %85+** : Phase 3+4+5 birlikte yüksek olasılıkla erişir.

---

# 🔧 Dosyalar — Hangi Yazılacak

| Dosya | Durum | Kim Yazacak |
|---|---|---|
| `generate_synthetic_data.py` | YAZILACAK | Phase 1 |
| `pretrain_synthetic.py` | YAZILACAK | Phase 2 |
| `finetune_iam.py` | YAZILACAK (greedy_aachen_v3.py'dan adapt) | Phase 3 |
| `train_ensemble.py` | YAZILACAK | Phase 4 |
| `ensemble_inference.py` | YAZILACAK | Phase 4 |
| `tta_inference.py` | YAZILACAK | Phase 5 |
| `greedy_aachen_v3.py` | MEVCUT (referans) | — |
| `trigram_lm.py` | MEVCUT (V3) | — |
| `word_beam_search.py` | MEVCUT | — |
| `_mcnemar_extended.py` | MEVCUT (analiz) | — |

---

# ⏱️ Süre Tahmini

| Aşama | Süre (Sıralı) | Süre (3 Makine Paralel) |
|---|---|---|
| 1. Synthetic data gen | 3-4h | 3-4h |
| 2. Synthetic pretrain | 4-6h | 4-6h |
| 3. IAM fine-tune (seed=42) | 3-4h | 3-4h |
| 4. Seed 123 + 2026 | 6-8h | 3-4h (paralel) |
| 5. TTA + ensemble inference | 30 min | 30 min |
| **TOPLAM** | **~18-22 saat** | **~14-16 saat** |

---

# ⚠️ Risk ve Önlemler

## Risk 1: Synthetic-Real Domain Gap
**Sorun**: Synthetic data printed fonts kullanır, gerçek el yazısı farklı görünür.
**Önlem**:
- El yazısı **simulating** fontlar seç (Caveat, Comic Sans, vb.)
- Aggressive augmentation (noise, blur, distortion)
- Fine-tune phase uzun olsun (40-50 epoch)

## Risk 2: Catastrophic Forgetting
**Sorun**: IAM fine-tune sırasında synthetic'ten öğrenilenler unutulur.
**Önlem**:
- Düşük LR (1e-4, peak 7e-4 değil)
- CNN'i ilk 5 epoch freeze et
- IAM train + sentetik mix de düşünülebilir

## Risk 3: Ensemble Overhead
**Sorun**: 3 model 3x disk + 3x inference latency.
**Önlem**:
- Resource-efficient claim'i ensemble için ÖZEL olarak gerekçelendir
- "Optional ensemble for accuracy-critical scenarios" diye yaz

## Risk 4: Reviewer "Niye Synthetic?" Diye Sorar
**Cevap**: 
> *"We pretrain on synthetic data to give the model exposure to a larger diversity
> of writer styles, since the IAM training set (747 forms) provides limited
> stylistic diversity. Synthetic data does not introduce vocabulary leak since
> the IAM Aachen test transcriptions are never used in synthetic data generation."*

---

# ✅ Başarı Kriterleri

Phase 5 sonunda **test set'inde**:

| Hedef Seviye | WA Eşiği | Aksiyon |
|---|---|---|
| ⭐⭐⭐ Mükemmel | **≥ %85.0** | Makale yazılır, hocaya sunulur |
| ⭐⭐ İyi | %82.5 - %84.9 | Makale yazılır, "approaching SOTA" framing |
| ⭐ Yeterli | %80.0 - %82.4 | Makale yazılır, "competitive baseline" framing |
| ⚠️ Yetersiz | < %80.0 | ResNet18 backbone denenir (Plan H3) |

Mevcut V3 zaten %78 → Phase 3 sonunda %81+ alırsak baseline'da güzel ilerleyiş var.

---

# 🚦 Şu An Ne Yapılacak

1. **Bu plan onaylanır**
2. **Synthetic data generation script (Phase 1) yazılır** ← İlk teknik iş
3. **Arkadaşına Phase 1+2+3 instructions verilir** (`ARKADAS_ICIN_H2.md`)
4. **Eğitim koşulur (Phase 2-5)**
5. **Sonuçlar analiz edilir, makale güncellenir**

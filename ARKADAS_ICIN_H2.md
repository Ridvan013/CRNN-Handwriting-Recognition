# Arkadaşın İçin H2 Görev Talimatı

Merhaba! Sen H2 planının teknik gelişimini üstleneceksin. Hedefimiz
%78.12'den %85+'a çıkmak. Bunun için **synthetic pretraining +
ensemble + TTA** yapacağız.

## ⚠️ Önce Bu İki Dosyayı OKU

1. **`KURALLAR.md`** — Metodoloji kuralları, yasaklar
   - Asla test set'ine dokunma
   - Cherry-pick yok
   - Her değişiklik için paired McNemar test
   - **MUTLAKA OKU yoksa baştan başlamamız gerekir**

2. **`H2_PLAN.md`** — H2'nin genel mantığı, 5 phase
   - Phase 1: Synthetic data gen
   - Phase 2: Synthetic pretrain
   - Phase 3: IAM fine-tune
   - Phase 4: Ensemble
   - Phase 5: TTA

---

## 🎯 Senin Ana Sorumluluğun

**Phase 1 + Phase 2 + Phase 3** — synthetic veri üretip pretrain yap,
sonra IAM Aachen üzerinde fine-tune et. Sonuç bir checkpoint olacak,
biz onunla ensemble ve TTA yapacağız.

---

## 📋 Phase 1 — Synthetic Data Generation

### 1.1 Kurulum

```bash
git clone https://github.com/Ridvan013/CRNN-Handwriting-Recognition.git
cd CRNN-Handwriting-Recognition
git checkout feature/aachen-v3-extended-trigram

# Synthetic data tool
pip install trdg

# Veya custom kullanacaksan:
pip install Pillow numpy opencv-python
```

### 1.2 Word Listesi Hazırla

```bash
python -c "
import nltk
nltk.download('words', quiet=True)
from nltk.corpus import words

# IAM train words
iam_words = []
with open('CRNN_1/aachen_splits/train_words.txt') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 9 and not line.startswith('#'):
            iam_words.append(parts[-1])

nltk_words = set(words.words())
all_words = list(set(iam_words) | nltk_words)
print(f'Total unique: {len(all_words):,}')

# Save for trdg
with open('synthetic_vocab.txt', 'w') as f:
    for w in all_words:
        f.write(w + '\n')
"
```

Beklenen çıktı: ~240,000 unique kelime, `synthetic_vocab.txt` dosyası.

### 1.3 Fontları İndir

**Google Fonts'tan indir** (ücretsiz, ticari kullanılabilir):

- Caveat: https://fonts.google.com/specimen/Caveat
- Indie Flower: https://fonts.google.com/specimen/Indie+Flower
- Kalam: https://fonts.google.com/specimen/Kalam
- Patrick Hand: https://fonts.google.com/specimen/Patrick+Hand
- Shadows Into Light: https://fonts.google.com/specimen/Shadows+Into+Light
- Architects Daughter: https://fonts.google.com/specimen/Architects+Daughter
- Permanent Marker: https://fonts.google.com/specimen/Permanent+Marker
- Sacramento: https://fonts.google.com/specimen/Sacramento
- Dancing Script: https://fonts.google.com/specimen/Dancing+Script
- Just Another Hand: https://fonts.google.com/specimen/Just+Another+Hand

Windows fontlarından da kullan:
- Segoe Script
- Comic Sans MS
- Lucida Handwriting
- Ink Free

Hepsini `fonts/` klasörüne koy. Toplam 15+ font olsun.

### 1.4 Synthetic Generation — trdg ile

**Önerilen tool**: TextRecognitionDataGenerator (trdg)

```bash
trdg \
  -i synthetic_vocab.txt \
  -c 300000 \
  --output_dir synthetic_data \
  -fd fonts/ \
  -k 5 \
  -rk \
  -bl 1 \
  -rbl \
  -do 1 \
  -or 1 \
  -wd 128 \
  -f 32 \
  -b 0 \
  -na 2 \
  -tc "#000000"
```

**Parametreler**:
- `-i`: vocabulary file
- `-c`: count (300,000 sample)
- `--output_dir`: output klasörü
- `-fd`: font directory
- `-k 5`: random skew (rotation ±5°)
- `-rk`: random skew enabled
- `-bl 1`: gaussian blur radius 1
- `-rbl`: random blur enabled
- `-do 1`: distorsion
- `-or 1`: orientation random
- `-wd 128`: width 128px
- `-f 32`: font size 32 (height ~32)
- `-b 0`: white background
- `-na 2`: name format word_count.png
- `-tc`: text color black

**Süre**: ~3-4 saat (CPU-bound, paralel çalıştırabilirsin)

### 1.5 Doğrulama

```bash
ls synthetic_data | head
ls synthetic_data | wc -l  # 300,000 olmalı
```

Görsel olarak 20-30 random sample aç, **el yazısı gibi mi** kontrol et.

### 1.6 Labels Dosyası

trdg otomatik olarak filename'i label yapar:
- `cat_0.png` → label "cat"
- `dog_125.png` → label "dog"

`generate_labels.py`:
```python
import os, re
labels = []
for fname in os.listdir("synthetic_data"):
    if fname.endswith(".png"):
        # Filename: "word_N.png" -> "word"
        word = re.sub(r'_\d+\.png$', '', fname)
        labels.append((fname, word))

with open("synthetic_data/labels.txt", "w") as f:
    for fname, label in labels:
        f.write(f"{fname} {label}\n")
print(f"Wrote {len(labels)} labels")
```

---

## 📋 Phase 2 — Synthetic Pretrain

### 2.1 Script: `pretrain_synthetic.py`

`greedy_aachen_v3.py`'dan başla, şunları değiştir:

**Data loading**:
```python
# Synthetic data loader yaz
class SyntheticDataset(Dataset):
    def __init__(self, labels_file, img_dir, transform=None):
        self.samples = []
        with open(labels_file) as f:
            for line in f:
                fname, label = line.strip().split(" ", 1)
                self.samples.append((os.path.join(img_dir, fname), label))
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 32))
        img_t = torch.from_numpy(img).float() / 255.0
        img_t = 1.0 - img_t
        img_t = (img_t - 0.5) / 0.5
        return img_t.unsqueeze(0), encode_label(label)
```

**Train/val split** (synthetic):
- 290,000 train, 10,000 val

**Hyperparameters**:
- `EPOCHS = 8`
- `BATCH_SIZE = 128`
- `LR = 1e-3`
- Cosine warmup 2 epoch + cosine decay
- AdamW, weight_decay=1e-5
- **Trigram correction kapalı** (synthetic için gereksiz)

### 2.2 Eğitim Süresi

300K sample × 8 epoch ÷ 128 batch = ~18,750 batch × 0.3s/batch = ~95 min = **~1.5-2 saat**

(Aslında ilk seferde 4-6 saat olur — kod hataları + GPU warmup vb.)

### 2.3 Checkpoint Kaydet

```python
torch.save(model.state_dict(), "Model_pretrain_synthetic/best_model_wa.pth")
```

Bu Phase 3'te kullanılacak.

### 2.4 Beklenen Synthetic Val WA

%90+ olmalı (synth data daha kolay).

Eğer %85'in altındaysa augmentation'lar çok agresif veya kod hatası var.

---

## 📋 Phase 3 — IAM Aachen Fine-tune

### 3.1 Script: `finetune_iam.py`

`greedy_aachen_v3.py`'yi kopyala. Şu değişiklikleri yap:

#### Değişiklik 1: Pretrained Checkpoint Yükle

```python
# Model oluştur (V3 architecture)
model = CRNNModel(img_height=32, img_width=128, num_classes=len(CHAR_LIST)+1)

# Pretrained checkpoint yükle
PRETRAIN_PATH = "Model_pretrain_synthetic/best_model_wa.pth"
if os.path.exists(PRETRAIN_PATH):
    print(f"Loading synthetic-pretrained weights from {PRETRAIN_PATH}")
    ckpt = torch.load(PRETRAIN_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt)
else:
    print("WARNING: No pretrained checkpoint found, training from scratch")

model = model.to(DEVICE)
```

#### Değişiklik 2: Daha Düşük LR

```python
# AdamW lr 7e-4 yerine 1e-4 (catastrophic forgetting önlemek için)
self.optimizer = optim.AdamW(
    self.model.parameters(),
    lr=1e-4,
    weight_decay=1e-5,
)
```

#### Değişiklik 3: model_dir

```python
model_dir = "Model_aachen_v3_pretrained"
```

#### Değişiklik 4: CNN Freeze (Opsiyonel ama Önerilen)

İlk 5 epoch boyunca CNN'i freeze et, synthetic'ten gelen visual features korunsun:

```python
# Phase 3 başında
for p in model.cnn.parameters():
    p.requires_grad = False

# trainer.train() içinde, her epoch başında:
if epoch == 5:
    print("Unfreezing CNN")
    for p in model.cnn.parameters():
        p.requires_grad = True
```

### 3.2 Eğitim

```bash
cd CRNN_1
python finetune_iam.py > finetune_iam.log 2>&1
```

Süre: ~3-4 saat (V3 ile aynı eğitim setup, IAM Aachen 31K sample × 40 epoch).

### 3.3 Beklenen Sonuç

| Metric | V3 (mevcut) | V3 + Pretrain (hedef) |
|---|---|---|
| Test WA (raw greedy) | 74.90% | **78-80%** |
| Test WA (greedy + V3 trigram) | 78.06% | **81-83%** |
| Test WA (beam + V3 trigram) | 78.12% | **81-83%** |

### 3.4 Validation

```bash
$env:CRNN_CSV = "Model_aachen_v3_pretrained/test_results_analysis.csv"
python _mcnemar_extended.py
```

**Önemli**: V3 (mevcut) vs V3-pretrained karşılaştırması paired McNemar
ile yapılır. Aynı test sample'larında her ikisinin tahminini al, b ve c
say, McNemar χ² hesapla. Anlamlı iyileşme p < 0.01 olmalı.

---

## 📋 Phase 4 — Ensemble (Sen mi Yapıyorsun?)

Eğer Phase 3 sonuçların ardından **%83+** alıyorsa, ensemble yapmaya
gerek olmayabilir. **%80-82** arası alırsan ensemble yap.

**Strateji**:
1. Aynı Phase 3 script'ini **3 farklı seed** ile koş:
   - `torch.manual_seed(42)`, save `Model_aachen_v3_pretrained_seed42/`
   - `torch.manual_seed(123)`, save `Model_aachen_v3_pretrained_seed123/`
   - `torch.manual_seed(2026)`, save `Model_aachen_v3_pretrained_seed2026/`
2. `ensemble_inference.py` yaz:
   ```python
   # 3 modeli yükle
   models = [load(f"Model_aachen_v3_pretrained_seed{s}/best_model_wa.pth") 
             for s in [42, 123, 2026]]
   
   # Test'te her image için:
   #   1. Her modelden log_probs al
   #   2. Average (softmax averaging)
   #   3. Greedy decode + V3 trigram correction
   ```

Toplam süre: 9 saat ardışık (3 × 3h) veya 3 saat paralel (3 makine).

---

## 📋 Phase 5 — TTA

Phase 4 bittikten sonra (veya Phase 3 sonunda eğer ensemble yoksa):

```python
def tta_predict(model, image):
    augs = [
        lambda x: x,
        lambda x: TF.rotate(x, 3),
        lambda x: TF.rotate(x, -3),
        lambda x: TF.affine(x, angle=0, translate=[0,0], scale=0.95, shear=0),
        lambda x: TF.affine(x, angle=0, translate=[0,0], scale=1.05, shear=0),
    ]
    log_probs_list = [model(aug(image).unsqueeze(0)) for aug in augs]
    return torch.mean(torch.stack(log_probs_list), dim=0)
```

Sonra greedy/beam decode + V3 trigram.

Beklenen: **+0.5-1.5pp** ek artış.

---

## 🆘 Takıldığında Sor

Karmaşık bir kısımda kalırsan, **direkt bana yaz**. Özellikle:

- Synthetic data görsel olarak çok printed görünüyorsa (handwriting değil)
- Synthetic val WA çok düşükse (kod hatası ihtimal)
- Fine-tune'da val WA artmıyorsa (LR çok düşük/yüksek)
- Beklenen pp artışı gerçekleşmiyorsa

## 🤖 Claude Yardımcı

Claude (Anthropic AI) ile çalışıyorum, sen de istersen onunla
konuşabilirsin. KURALLAR.md ve H2_PLAN.md'yi context olarak verirsen
metodolojiyi tam anlar.

## 📦 Bana Bildireceklerin

Phase her bitince:

1. **Test WA** (kesin sayı)
2. **Wilson 95% CI**
3. **McNemar p-value** (V3-mevcut vs V3-pretrained)
4. **Eğitim log dosyası**
5. **`test_results_analysis.csv`**

Bu bilgilerle bir sonraki phase'e mi geçeceğimize karar veririz.

## 🎯 Ana Hedef Hatırlatma

**%85+ Aachen test WA**

Eğer Phase 3 sonunda %83+ alırsan zaten hedefe yakınız. Phase 4/5 +1-2pp
katar ve %85'i geçeriz.

İyi şanslar! 🚀

---

## Hızlı Referans — Komutlar

```bash
# Phase 1
python -c "from nltk.corpus import words; ..."  # vocab
trdg -i synthetic_vocab.txt -c 300000 ...

# Phase 2
python pretrain_synthetic.py > pretrain.log 2>&1

# Phase 3
python finetune_iam.py > finetune.log 2>&1

# Phase 4
for $seed in 42 123 2026:
    python finetune_iam_seed.py --seed $seed > finetune_$seed.log

# Phase 5 / Ensemble inference
python ensemble_inference.py
python tta_inference.py
```

Sorularını bekliyorum.

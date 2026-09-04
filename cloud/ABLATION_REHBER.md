# Ablation Deneyleri — Çalıştırma Rehberi

> ## ⚠️ GÜNCELLEME (tam veri)
> Bu rehberdeki 78.06 / 84.54 / N=5,338 sayıları IAM'in **%39'luk kesik** bir
> etiket dosyasıyla elde edilmişti (44,859 / 115,320 kayıt; test 336 formun
> yalnızca 87'sinde). Repo artık **tam IAM etiketlerinden** kurulmuş split
> dosyalarını taşıyor:
>
> | Split | Form | Kelime (ok) | Eski (kesik) |
> |---|---:|---:|---:|
> | train | 747 | **47,999** | 31,615 |
> | validation | 116 | **7,559** | 1,646 |
> | test | 336 | **20,310** | 5,338 |
>
> Sonuç: **baseline dahil her şey tam veriyle yeniden eğitilir.** Notebook
> (`ablation_kaggle.ipynb`) iki oturuma bölündü: **A** = `narrow` + `full` +
> lexicon ablation (~7.5 s), **B** = `photo` + `elastic` + `morph` (~10 s).
> Hücre 1'deki `SESSION` değişkenini seç, Save & Run All. Kaggle'daki
> `words.txt` artık kullanılmıyor, yalnız görüntüler.
>
> Aşağıdaki bölümler mantığı anlatmak için duruyor; sayılar eski.


Bu dosya, makaleye eklenecek **iki ablation tablosunu** üretmek için ne
yapılacağını anlatır. Kod hazır; yapılması gereken tek şey Kaggle'da bir
notebook çalıştırmak.

---

## 1. Neden bu deneyleri yapıyoruz?

Hocanın revizyon notu:

> *"elastic tek başına ne kadar arttırdı, morfolojik ne kadar arttırdı,
> augmentation'da bunları net olarak belirtip göstermemiz lazım"*
>
> *"trigramın veya diğer her şeyin sırasıyla ne kadar acc'i artırdığını
> göstermemiz lazım"*

Şu an makalede sadece **iki nokta** var:

```
augmentation yok       →  78.06 %
elastik + morfolojik   →  84.54 %
                          ─────────
                          +6.48 puan
```

Ama bu 6.48 puanın **ne kadarı elastikten, ne kadarı morfolojikten** geliyor
bilmiyoruz. Hoca bunu istiyor. Aradaki basamakları ölçmek için yeni eğitimler
gerekiyor — uydurma sayı yazamayız.

---

## 2. Ablation mantığı: her seferinde TEK değişken

Ablation'ın kuralı şudur: iki satır arasında **sadece bir şey** değişmeli.
Yoksa farkın neyden geldiği belirsiz kalır.

### Tablo A — Augmentation ablation

| # | Konfigürasyon | Geniş fotometrik | Elastik | Morfolojik | Durum |
|---|---|:---:|:---:|:---:|---|
| 1 | `CRNN-L` | ✗ | ✗ | ✗ | **var** (78.06) |
| 2 | `+ wide photometric` | ✓ | ✗ | ✗ | **eğitilecek** |
| 3 | `+ elastic` | ✓ | ✓ | ✗ | **eğitilecek** |
| 4 | `+ morphological` | ✓ | ✗ | ✓ | **eğitilecek** |
| 5 | `AugCRNN-T` | ✓ | ✓ | ✓ | **var** (84.54) |

Okuma biçimi:

- **1 → 2**: fotometrik aralığı genişletmenin katkısı
- **2 → 3**: elastik deformasyonun **tek başına** katkısı
- **2 → 4**: morfolojik bozulumun **tek başına** katkısı
- **3, 4 → 5**: ikisinin birlikte kullanılmasının ek katkısı

2. satır neden gerekli? Çünkü mevcut `CRNN-L` dar fotometrik aralık kullanıyor
(0.85–1.15), `AugCRNN-T` ise geniş (0.70–1.35). Bu satır olmasaydı elastik'in
katkısı fotometrik değişimle karışırdı ve "elastik tek başına ne yaptı?"
sorusunu cevaplayamazdık.

### Tablo B — Lexicon / trigram ablation

| # | Konfigürasyon | Sözlük | Trigram skorlama |
|---|---|---|:---:|
| 1 | `AugCRNN` | yok | ✗ |
| 2 | `+ IAM lexicon` | IAM 5.9K | ✗ |
| 3 | `+ IAM lexicon + trigram` | IAM 5.9K | ✓ |
| 4 | `AugCRNN-T` (önerilen) | IAM + NLTK 238K | ✓ |

- **1 → 2**: sözlük kontrolünün katkısı
- **2 → 3**: n-gram skorlamasının katkısı
- **3 → 4**: sözlüğü 5.9K'dan 238K'ya genişletmenin katkısı

**Bu tablo için eğitim GEREKMİYOR.** Dördü de aynı modelin aynı çıktısına
farklı post-processing uygulanmasıyla elde ediliyor. Model bir kez çalışıyor,
hipotezler saklanıyor, sonra dört farklı düzeltme uygulanıyor.

---

## 3. Neden Kaggle'da, neden yerelde değil?

Hocanın diğer notu:

> *"farklı ortamlarda denendi, optimum ortamın sadece bilgileri verilsin,
> kıyasa girmeyelim"*

Aynı model yerelde çalıştırıldığında 84.54 yerine 78.29 çıkıyor (NumPy/cuDNN/GPU
farkları). Eğer bazı satırları Kaggle'da bazılarını yerelde ölçersek tablo
kendi içinde tutarsız olur.

**Kural: bütün satırlar aynı ortamda, yani Kaggle T4'te ölçülmeli.**

---

## 4. Kaggle'da çalıştırma (adım adım)

### 4.1 Notebook'u indir

```
https://github.com/Ridvan013/CRNN-Handwriting-Recognition/raw/feature/aachen-v3-extended-trigram/cloud/ablation_kaggle.ipynb
```

Tarayıcı JSON gösterirse `Ctrl+S` ile `.ipynb` uzantısıyla kaydet.

### 4.2 Kaggle'a yükle

`+ Create` → `New Notebook` → `File` → `Import Notebook` → Upload.

### 4.3 İki input ekle

Sağ panelden `+ Add Input`:

1. **IAM word dataset** — `words.txt` ve `words/` klasörü içeren herhangi biri
   (örn. `iam_handwriting_word_database`)
2. **Eğitilmiş AugCRNN-T ağırlıkları** — `berhat-v3-augmented-model`
   (Tablo B bunu kullanıyor; yoksa Tablo B atlanır, Tablo A yine çalışır)

### 4.4 Ayarlar

| Ayar | Değer |
|---|---|
| Accelerator | **GPU T4** |
| Internet | **ON** (git clone + NLTK indirmesi için) |
| Persistence | Files only |

### 4.5 Çalıştır

`Save Version` → **`Save & Run All (Commit)`** → Save.

Quick Save **değil** — o arka planda çalıştırmaz.

---

## 5. Notebook ne yapıyor? (hücre hücre)

| Hücre | İş | Süre |
|---|---|---|
| 1 | GitHub'dan repoyu çeker, NLTK'yi indirir | ~1 dk |
| 2 | IAM veri yolunu ve model ağırlıklarını otomatik bulur | ~3 dk |
| 4 | **Tablo B** — lexicon ablation (eğitim yok) | ~15 dk |
| 6 | **Tablo A / 1** — `--aug-mode photo` eğitimi | ~2 saat |
| 7 | **Tablo A / 2** — `--aug-mode elastic` eğitimi | ~2 saat |
| 8 | **Tablo A / 3** — `--aug-mode morph` eğitimi | ~2 saat |
| 9 | İki tabloyu da derleyip ekrana basar | anlık |

**Toplam ~6.5 saat.** Kaggle oturum limiti 12 saat, rahat sığıyor.

Notebook'u başlattıktan sonra sekmeyi kapatabilirsin, PC'yi kapatabilirsin —
Kaggle sunucuda çalışmaya devam eder, bitince e-posta gelir.

---

## 6. Manuel çalıştırmak istersen (notebook olmadan)

### Tablo B — lexicon ablation

```bash
python cloud/ablation_lexicon.py \
    --model      Model_aachen_v3_augmented/best_model_wa.pth \
    --iam-words  /kaggle/input/.../words.txt \
    --iam-root   /kaggle/input/.../words \
    --out        results/ablation_lexicon.json
```

### Tablo A — üç eğitim

```bash
# 1/3  geniş fotometrik, elastik YOK, morfolojik YOK
python cloud/v3_augmented_train.py --aug-mode photo \
    --epochs 100 --batch 128 --lr 7e-4 --patience 15 \
    --model-dir abl_photo \
    --iam-words <words.txt> --iam-root <words/>

# 2/3  geniş fotometrik + elastik  (morfolojik YOK)
python cloud/v3_augmented_train.py --aug-mode elastic \
    --epochs 100 --batch 128 --lr 7e-4 --patience 15 \
    --model-dir abl_elastic \
    --iam-words <words.txt> --iam-root <words/>

# 3/3  geniş fotometrik + morfolojik  (elastik YOK)
python cloud/v3_augmented_train.py --aug-mode morph \
    --epochs 100 --batch 128 --lr 7e-4 --patience 15 \
    --model-dir abl_morph \
    --iam-words <words.txt> --iam-root <words/>
```

**Değişen tek şey `--aug-mode`.** Diğer bütün hiperparametreler
(epoch, batch, learning rate, patience, seed) sabit — ablation'ın kuralı bu.

### `--aug-mode` seçenekleri

| Mod | Geniş fotometrik | Elastik | Morfolojik | Karşılığı |
|---|:---:|:---:|:---:|---|
| `full` | ✓ | ✓ | ✓ | **AugCRNN-T (varsayılan)** |
| `elastic` | ✓ | ✓ | ✗ | Tablo A satır 3 |
| `morph` | ✓ | ✗ | ✓ | Tablo A satır 4 |
| `photo` | ✓ | ✗ | ✗ | Tablo A satır 2 |
| `narrow` | ✗ | ✗ | ✗ | CRNN-L (baseline) |

`--aug-mode` yazmazsan `full` çalışır, yani eski davranış hiç değişmedi.

---

## 7. Bittiğinde ne yapmalı?

Kaggle **Output** sekmesinden şunları indir:

```
results/ablation_lexicon.json          ← Tablo B
abl_photo/test_results_analysis.csv    ← Tablo A satır 2
abl_elastic/test_results_analysis.csv  ← Tablo A satır 3
abl_morph/test_results_analysis.csv    ← Tablo A satır 4
```

Ayrıca 9. hücrenin ekran çıktısını da gönder — iki tablo orada derlenmiş
halde yazdırılıyor.

Bu dosyalar geldiğinde iki tablo makaleye eklenecek.

---

## 8. Sık karşılaşılan sorunlar

| Belirti | Sebep / çözüm |
|---|---|
| `IAM dataset bulunamadi` | Add Input'tan IAM word dataset eklenmemiş |
| `AugCRNN-T agirliklari: None` | Model dataset'i eklenmemiş → Tablo B atlanır, Tablo A çalışır |
| `git clone` hatası | Settings → Internet **ON** değil |
| Eğitim çok yavaş | Accelerator `None` kalmış, GPU seçilmemiş |
| Oturum 12 saatte kesildi | Hücre 6/7/8'i ayrı Save Version'larda çalıştır |

---

## 9. Özet

- **Değişen tek şey:** `--aug-mode` bayrağı
- **Sabit kalan:** mimari, epoch, batch, lr, patience, seed, veri
- **Amaç:** 6.48 puanın hangi tekniğe ait olduğunu göstermek
- **Süre:** ~6.5 saat (Tablo B ilk 15 dakikada hazır)
- **Kritik:** hepsi aynı ortamda (Kaggle T4) ölçülmeli

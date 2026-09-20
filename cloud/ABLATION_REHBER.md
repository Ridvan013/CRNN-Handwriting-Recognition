# Ablation Deneyleri — Çalıştırma Rehberi

> ## ✅ TAMAMLANDI (7 Eylül 2026)
> Beş konfigürasyon tam veriyle, tek makinede, tek kod sürümüyle eğitildi ve
> ölçüldü. Sonuçlar ve tüm tablolar: **`results/ABLATION_SONUC.md`**.
> Makale bu sayılarla yeniden yazıldı (`makale/paper.tex`). Aşağısı, deneyleri
> yeniden koşmak isteyen için kalıyor.

> ## ⚠️ GÜNCELLEME (tam veri)
> Bu rehberdeki 78.06 / 84.54 / N=5,338 sayıları IAM'in **%39'luk kesik** bir
> etiket dosyasıyla elde edilmişti (44,859 / 115,320 kayıt; test 336 formun
> yalnızca 87'sinde). Repo artık **tam IAM etiketlerinden** kurulmuş split
> dosyalarını taşıyor:
>
> | Split | Form | Kelime (ok) | Eski (kesik) |
> |---|---:|---:|---:|
> | train | 747 | **47,999** | 31,615 |
> | validation | 111 | **7,205** | 1,646 |
> | test | 336 | **20,310** | 5,338 |
>
> Doğrulama 116 yerine 111 form: metni test kümesinde de geçen 5 form
> (`f07-028b`, `f07-032b`, `f07-039b`, `f07-042b`, `f07-046b`) doğrulamadan
> çıkarıldı; en iyi epoch seçimi test'ten bağımsız olsun diye. **Test kümesine
> dokunulmadı**, literatür karşılaştırması etkilenmiyor.
>
> Sonuç: **baseline dahil her şey tam veriyle yeniden eğitilir.** Notebook
> (`ablation_kaggle.ipynb`) iki oturuma bölündü: **A** = `narrow` + `full` +
> lexicon ablation (~7.5 s), **B** = `photo` + `elastic` + `morph` (~10 s).
> Hücre 1'deki `SESSION` değişkenini seç, Save & Run All. Kaggle'daki
> `words.txt` artık kullanılmıyor, yalnız görüntüler.
>
> Aşağıdaki bölümler mantığı anlatmak için duruyor; sayılar eski.


> ## ⚠️ GÜNCELLEME 2 — GPU hattı ve elastik bulgusu
>
> **Hız.** Augmentation artık `cloud/gpu_aug.py` ile toplu olarak GPU'da yapılıyor;
> veri kümesi tek bir uint8 tensor olarak GPU'da duruyor. 128'lik batch'in
> augmentation'ı 5–18 ms (epoch başına 2–7 s). Eski per-image yol yerelde epoch
> başına 10–50 dk sürüyordu. Eski yol `--gpu-aug 0` ile hâlâ seçilebilir.
> Görüntüler bir kez 64×256'ya getirilip `cache/` altına yazılır; sonraki
> koşularda yükleme saniyeler sürer (`--no-cache` ile kapatılır).
>
> **Elastik deformasyon aslında çalışmıyormuş.** Orijinal `_elastic_deform`,
> normalize edilmiş Gaussian-blur'lu gürültüyü α∈[2,5] ile çarpıyor; blur'un
> genliği ~0.01 olduğu için piksel yer değiştirmesi **~0.03–0.06 px RMS**
> (α=5'te en fazla 0.19 px). Yani "kalem titremesi" dönüşümü fiilen no-op.
> Bu, "elastic tek başına katkı yapmadı" bulgusunu açıklıyor.
>
> Düzeltilmiş parametrizasyon eklendi: `--elastic-legacy-amplitude 0` ile
> α doğrudan **piksel cinsinden RMS yer değiştirme** olur (`--elastic-alpha 1 3`
> önerilir; Simard 2003 ölçeğinde). Varsayılan hâlâ legacy (sadık yeniden
> üretim). Hangi genlikle koşulacağı bir **araştırma kararıdır**, hocaya
> sorulmalı. Tüm konfigürasyonlar aynı seçimle koşulmalı.
>
> ```powershell
> .
un_ablation.ps1 main                                        # legacy (no-op elastik)
> .
un_ablation.ps1 main -ElasticLegacy 0 -ElasticAlpha "1 3"    # düzeltilmiş
> ```

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

---

## 10. Seed tekrarları (hoca revizyonu, 15 Eylül) — **TAMAMLANDI (20 Eylül)**

> Berhat dört eğitimi Kaggle'da yaptı, ağırlıkları `brht25/seed1-output` ve
> `brht25/seed2-output` veri setleriyle paylaştı; puanlama yapıldı. Sonuç:
> fark seedler arası işaret değiştiriyor (+0,32 / −0,49 / +0,78 pp,
> ortalama +0,20 ± 0,64), makalede Tablo 3. Ayrıntı:
> `results/ABLATION_SONUC.md` → "Seed tekrarları". Aşağıdaki talimatlar
> tekrar gerekirse diye duruyor.

### Neden?

Tablo 2'deki her konfigürasyon **bir kez** eğitildi (seed 42). CRNN-B'den
CRNN-LX'e +0,38 puanlık farkın augmentation'dan mı, yoksa ağırlıkların
rastgele başlangıcından mı geldiğini tek seed ile ayırt edemeyiz. Hoca
en azından **CRNN-B ve CRNN-LX'i 3 seed ile** koşup ortalama ± standart
sapma raporlamamızı istedi. Seed 42 zaten var; **4 yeni eğitim** gerekiyor:

| Koşu | `--aug-mode` | `--seed` | `--model-dir` |
|---|---|---|---|
| 1 | `narrow` (CRNN-B) | 123 | `Model_seed_narrow_123` |
| 2 | `narrow` (CRNN-B) | 456 | `Model_seed_narrow_456` |
| 3 | `full` (CRNN-LX) | 123 | `Model_seed_full_123` |
| 4 | `full` (CRNN-LX) | 456 | `Model_seed_full_456` |

**Seed dışında hiçbir şey değişmiyor.** Aynı veri, aynı bölme, aynı epoch,
batch, lr, patience, aynı düzeltilmiş elastik ayarı.

### Gereklilikler

1. **Repo, güncel dal:** `feature/aachen-v3-extended-trigram` — `git pull`
   yap; 15 Eylül commit'i şart (UTF-8 çıktı düzeltmesi, `--aug-mode none`,
   `ablation_lexicon_all.py`'ye `ad=dizin` desteği bu commit'te).
2. **Etiketler** repodan gelir: `aachen_splits/{train,validation,test}_words.txt`
   (tam IAM, 47.997 / 7.205 / 20.310). Ekstra bir şey indirmene gerek yok.
3. **Görüntüler:** Kaggle'daki IAM word dataset'i (§4.3'teki gibi Add Input).
   `--iam-root` o dataset'in `words/` klasörünü, `--iam-words` da içindeki
   `words.txt`'yi gösterecek. Bu dosyanın eksik (44.859 satırlık) kopyası
   olması sorun değil; etiketler zaten repodan okunuyor.
4. **NLTK words** corpus'u (Internet ON; script kendisi indirir).
5. **GPU:** T4 yeterli. Bir eğitim T4'te yaklaşık 3–5 saat (100 epoch'a
   kadar, early stopping ile genelde 70–100 epoch). Kaggle oturumu 12 saat:
   **bir oturuma 2 eğitim** sığar → toplam **2 oturum**. Haftalık 30 saat GPU
   kotasına dikkat.

### Komutlar (repo kökünde)

```bash
# oturum 1
python cloud/v3_augmented_train.py --aug-mode narrow --seed 123 \
    --model-dir Model_seed_narrow_123 \
    --epochs 100 --batch 128 --lr 7e-4 --patience 15 \
    --elastic-legacy-amplitude 0 --elastic-alpha 1 3 \
    --iam-words <words.txt> --iam-root <words/>

python cloud/v3_augmented_train.py --aug-mode full --seed 123 \
    --model-dir Model_seed_full_123 \
    --epochs 100 --batch 128 --lr 7e-4 --patience 15 \
    --elastic-legacy-amplitude 0 --elastic-alpha 1 3 \
    --iam-words <words.txt> --iam-root <words/>

# oturum 2: aynı iki komut, --seed 456 ve ..._456 dizinleriyle
```

Notebook kullanıyorsan (§4), 6/7/8. hücrelerdeki komut listesini bu dört
komutla değiştirmen yeterli; `SESSION` mantığı aynen çalışır.

### Bittiğinde ne göndereceksin?

Her `Model_seed_*` klasöründen:

```
best_model_wa.pth          (~115 MB, asıl gereken bu)
training_history.json
results.json
test_results_analysis.csv
```

`.pth` dosyaları git'e sığmaz: dört klasörü Kaggle Dataset olarak yayınla
ya da Drive'a koy, linki gönder. Puanlama Rıdvan'ın makinesinde, makaledeki
diğer altı modelle **aynı deterministik yoldan** yapılacak:

```bash
python cloud/ablation_lexicon_all.py \
    --modes narrow,narrow_s123=Model_seed_narrow_123,narrow_s456=Model_seed_narrow_456,full,full_s123=Model_seed_full_123,full_s456=Model_seed_full_456 \
    --mcnemar-baseline narrow \
    --out results/ablation_seeds.json --dump-preds results/preds_seeds \
    --iam-words <words.txt> --iam-root <words/>
```

Makaledeki Tablo 2 artık **trigram + tüm-satır (Viterbi) çözücü**yle
puanlanıyor (`cloud/ablation_viterbi.py`, aynı `ad=dizin` sözdizimi; `full`
listede olmalı çünkü çözücü seçimi onun doğrulamasında yapılıyor); seed
modelleri de aynı script'le puanlanacak:

```bash
python cloud/ablation_viterbi.py \
    --modes narrow,narrow_s123=Model_seed_narrow_123,narrow_s456=Model_seed_narrow_456,full,full_s123=Model_seed_full_123,full_s456=Model_seed_full_456 \
    --baseline narrow \
    --out results/ablation_viterbi_seeds.json --dump-preds results/preds_viterbi_seeds \
    --iam-words <words.txt> --iam-root <words/>
```

(NLTK `words` ve `brown` corpus'ları gerekir; script ilk çalıştırmada
`nltk.download` ile indirir, Internet açık olmalı.)

Çıkan üçer WA'dan ortalama ± SD hesaplanıp Tablo 2'ye hocanın istediği
satırlar eklenecek. Sonuç iki yöne de çıkabilir: fark seedler arası
oynamanın içinde kalırsa "anlamlı değil" bulgusu güçlenir; +0,4 puan her
seedde tekrar ederse "küçük ama tutarlı" diye yazılır. İkisi de dürüst
sonuçtur, hangisi çıkarsa o.

### Dikkat

- Eğitim script'inin sonunda basılan `SONUÇ` bloğu ve `results.json`'daki
  `greedy_trigram_wa_pct`, **eski düzelticiyle (unigram) ve fp16 ile**
  hesaplanır; makaledeki sayılar değildir. Sadece "eğitim bitti mi" kontrolü
  için bak. Makaleye giren sayılar yukarıdaki puanlama komutlarından çıkar.
  Eğitim kodunu **değiştirme**: checkpoint seçimi ve early stopping seed 42
  modellerinde nasıl yapıldıysa seedlerde de aynı olmalı, yoksa karşılaştırma
  bozulur.
- `--seed` bayrağını **her komutta** ver; vermezsen 42 ile eğitir ve mevcut
  sonucu tekrarlamış olursun.
- `--model-dir` adlarını aynen kullan; puanlama komutu bu adlara göre.
- Eğitim `Early stopping at epoch N` yazıp `SONUÇ` bloğunu basmadan biterse
  oturum kesilmiştir; `training_history.json` kaç epoch gittiğini gösterir,
  o koşuyu baştan al.

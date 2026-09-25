KAPSAM

Rapor üç soruyu cevaplıyor: (a) elastic ve morfolojik augmentation'ın ayrı ayrı
katkısı nedir, (b) yayınlanan 84.54% word accuracy'nin bileşenleri nelerdir,
(c) ölçümlerin yapıldığı veri seti makalede tarif edilen veri seti mi. İkinci ve
üçüncü sorular ilk plana dahil değildi; birinci sorunun ölçümü sırasında ortaya
çıktılar ve ikisi de makalenin ana iddialarını doğrudan etkiliyor.

YÖNTEM

Beş eğitim karşılaştırıldı, hepsi aynı script (cloud/v3_augmented_train.py), aynı
hiperparametreler (lr 7e-4, batch 128, patience 15), sadece --aug-mode değişiyor.
İki değerlendirme yapıldı: ham greedy CTC çıktısı (sözlük/trigram yok) ve trigram
corrector sonrası. Lexicon ablation'ı tek modelin cache'lenmiş hipotezleri
üzerinde dört post-processing konfigürasyonu ile yürütüldü, böylece farklar
yalnızca post-processing'i yansıtıyor. Tüm run'ların per-sample CSV'leri birebir
hizalı (5,338 satır, 0 ground-truth uyuşmazlığı), bu yüzden eşleşmiş McNemar
testleri uygulanabildi.

TABLO 1 — AUGMENTATION ABLATION, HAM GREEDY (sözlük yok)

| Konfigürasyon                  |    WA |   CER | 95% CI         |     Δ |
|--------------------------------|-------|-------|----------------|-------|
| narrow (augmentation yok)      | 73.15 | 11.71 | [71.95, 74.33] |     — |
| wide photometric               | 73.55 | 11.38 | [72.35, 74.71] | +0.39 |
| photo + elastic (37 ep)        | 71.96 | 12.33 | [70.74, 73.14] | −1.20 |
| photo + elastic (rerun, 55 ep) | 73.55 | 11.48 | [72.35, 74.71] | +0.39 |
| photo + morph                  | 72.46 | 12.08 | [71.25, 73.64] | −0.69 |
| AugCRNN-T (üçü birden)         | 74.97 | 10.83 | [73.79, 76.12] | +1.82 |

Photo tabanına göre: elastic +0.00, morfolojik −1.09, üçü birlikte +1.42.

Elastic'in ilk turdaki düşük değeri 37 epoch'ta early-stop'tan kaynaklanıyormuş;
patience 25 ile tekrarlandığında (55 epoch) photo ile birebir aynı kelime sayısını
tutturuyor (3926/5338). Yani elastic'in ölçülebilir bir katkısı yok. Morfolojik
augmentation photo üstüne eklendiğinde zarar veriyor. Sadece üçünün kombinasyonu
net pozitif.

Güven aralıkları geniş ölçüde örtüşüyor. Ancak bu, farkların anlamsız olduğu
anlamına gelmiyor: aşağıdaki eşleşmiş testler, örtüşen CI'lara rağmen tam
augmentation'ın katkısının anlamlı, elastic'in katkısının ise tam olarak sıfır
olduğunu gösteriyor.

TABLO 2 — AYNI RUN'LAR, TRIGRAM CORRECTOR SONRASI

| Konfigürasyon             |    WA |   CER | epoch |
|---------------------------|-------|-------|-------|
| narrow (augmentation yok) | 83.12 |  9.82 |    47 |
| wide photometric          | 83.05 |  9.91 |    60 |
| photo + elastic           | 83.01 | 10.14 |    37 |
| photo + elastic (rerun)   | 83.18 |  9.82 |    55 |
| photo + morph             | 82.50 | 10.13 |    56 |
| AugCRNN-T                 | 84.54 |  9.24 |     — |

Augmentation'ı tamamen kapalı baseline 83.12 alıyor. Makale baseline'ı 78.06
olarak veriyor. Aynı script ve aynı corrector ile ölçüldüğünde gerçek fark
83.12 → 84.54, yani +1.42 pp; iddia edilen +6.48 pp değil.

DİKKAT: bu tablodaki corrector tüm words.txt'ten kuruluyor, yani her satırda
test-set sızıntısı var. Tablo yalnızca kendi içinde karşılaştırma için geçerli.

EŞLEŞMİŞ ANLAMLILIK TESTLERİ (exact McNemar)

Per-sample CSV'ler aynı sırada ve aynı ground-truth'a sahip olduğu için makalenin
kendi kullandığı test uygulanabiliyor:

| Karşılaştırma               |   b |   c |       p |
|-----------------------------|-----|-----|---------|
| AugCRNN-T vs narrow         | 308 | 232 |  0.0012 |
| AugCRNN-T vs elastic (rerun)| 303 | 230 |  0.0018 |
| elastic (rerun) vs narrow   | 288 | 285 |  0.93   |

İki sonuç çıkıyor. Birincisi: augmentation'ın +1.42 pp'lik katkısı, küçülmüş
haliyle bile p < 0.01 düzeyinde anlamlı. Makalenin merkezi iddiası tamamen
çökmüyor, 4.5 kat küçülüyor. İkincisi: elastic'in katkısı sadece "ölçülemedi"
değil, p = 0.93 ile tam anlamıyla sıfır — iki model neredeyse aynı örnekleri
doğru ve yanlış yapıyor.

Sınır: McNemar, bu iki spesifik modelin bu test setinde farklı olduğunu söyler;
augmentation'ın tekrar edilebilir biçimde +1.42 pp getirdiğini söylemez. Seed
varyansı hâlâ ölçülmedi.

TABLO 3 — LEXICON / TRIGRAM ABLATION

| Konfigürasyon              | Temiz sözlük (train-split) | Sızıntılı sözlük (tüm words.txt) |
|----------------------------|---------------------------|----------------------------------|
| ham greedy, sözlük yok     | 74.97                     | 74.97                            |
| IAM lexicon                | 74.48                     | 82.54                            |
| IAM lexicon + trigram      | 72.89                     | 84.58                            |
| NLTK genişletme            | 77.97                     | 83.74                            |
| sözlük boyutu (IAM/+NLTK)  | 6,339 / 238,798           | 14,294 / 242,609                 |

İki gözlem. Birincisi: temiz kurulumda küçük IAM sözlüğü modeli bozuyor — WA
74.97'den 72.89'a düşerken CER 10.83'ten 16.11'e çıkıyor. Corrector doğru
tahminleri zorla sözlük içine çekiyor. NLTK genişletmesi ise bunu düzeltip
+3.00 pp getiriyor. Yani makalenin §III-E'deki argümanı — "5.9K'lık listeye
kısıtlı bir corrector, sırf listede yok diye geçerli hipotezleri reddeder" —
sızıntısız kurulumda tam olarak doğrulanıyor. Sözlük genişletmesi, sızıntı
temizlendikten sonra makalede ayakta kalan tek post-processing katkısı.

İkincisi: sızıntılı kurulumda NLTK genişletmesi zarar veriyor (84.58 → 83.74),
çünkü sızıntılı IAM sözlüğü zaten test kelimelerini içeriyor ve genişletme
yalnızca gürültü ekliyor. Bu, "önerilen sözlük kötü" demek değil; "yayınlanan
84.54 konfigürasyonu zaten sakat" demek.

SIZINTI İZOLASYON DENEYİ

Sızıntılı ve temiz sözlük iki bakımdan farklı: biri test formlarının kelimelerini
içeriyor, diğeri iki kat daha büyük. Hangisinin baskın olduğunu ayırmak için
sözlük boyutu neredeyse sabit tutulup sadece test formları çıkarıldı.

| Sözlük                     | Boyut  |    WA |
|----------------------------|--------|-------|
| train-split                |  6,339 | 72.89 |
| test formları hariç, büyük | 13,518 | 74.09 |
| tüm words.txt              | 14,294 | 84.58 |

776 test kelimesinin çıkarılması WA'yı 10.49 pp düşürüyor. Sözlük boyutunun
katkısı yalnızca +1.20 pp. Sonuç kesin: kazanç boyuttan değil, test-set
sızıntısından geliyor.

Sızıntı tavanı bağımsız olarak da doğrulandı: 5,338 test token'ının ground truth
kelimesi train sözlüğünde olmayan ama sızıntılı sözlükte olan kısmı 953 token,
yani %17.85 (ilk hesapta 913 / %17.1 bulunmuştu; fark yalnızca kelime
normalizasyonundan kaynaklanıyor, sonucu değiştirmiyor). Bu 953 token'ın
tamamı tüm-words.txt sözlüğünde mevcut. Ölçülen +6.6 pp'lik kazanç bu tavanın
içinde.

CORRECTOR KARŞILAŞTIRMASI

Eski CRNN-L modelinin kaydedilmiş ham CTC çıktısına iki farklı post-processing
yığını uygulandı. Model ve optik çıktı birebir aynı.

| Konfigürasyon                        |      WA |   CER |
|--------------------------------------|---------|-------|
| ham greedy                           |   74.90 | 10.82 |
| eski yığın (makaledeki 78.06)        |   78.06 | 11.16 |
| yeni yığın (ablation pipeline)       |   84.28 |  9.18 |

Aynı model üzerinde post-processing değişikliği tek başına +6.22 pp getiriyor.
Makalenin augmentation'a atfettiği +6.48 pp ile bu neredeyse aynı büyüklükte.

Not: bu +6.22 pp, implementasyon değişikliği (trigram_lm.py → trigram_lm2.py) ile
sözlük kaynağı değişikliğini aynı anda içeriyor; ikisi ayrıştırılmadı. O yüzden
"corrector kodu +6.22 getirdi" değil, "post-processing yığını +6.22 getirdi"
denmeli. Sonucu değiştirmiyor: her iki bileşen de augmentation değil.

VERİ SETİ: words.txt KESİK

Bu, raporun ilk sürümünde dipnot olarak geçiyordu; incelendiğinde sızıntıdan daha
ağır bir sorun olduğu görüldü.

archive/iam_words/words.txt yalnızca a01–e07 form öneklerini içeriyor; dosya
alfabenin ortasında kesilmiş. 613 form, 51,615 kayıt. Buna karşılık görüntü
dizini eksiksiz: 1,616 form klasörü, 115,320 PNG. Yani eksik olan annotation
dosyası, veri değil.

Sonuçlar:

| Ölçü                              | Kullanılan | Gerçek Aachen |
|-----------------------------------|------------|---------------|
| train formu                       |        498 |           747 |
| validation formu                  |         25 |           116 |
| test formu                        |         87 |           336 |
| test kelimesi                     |      5,338 |             — |

Test seti, Aachen test setinin %26'sı. f–r arası form öneklerine sahip yazarlar
ne train'de ne test'te var. Makalenin §III-A'da verdiği 31,320 / 1,646 / 5,338
sayıları "yalnızca ok işaretli crop'lar tutuldu" ile açıklanıyor; gerçek sebep
bu değil, kesik annotation dosyası. ok filtresi tek başına ~96K kayıt bırakır.

Bu, iç karşılaştırmaları (Tablo 1, 2, 3) etkilemiyor — tüm run'lar aynı dosyayı
kullandı. Ama makalenin en güçlü savını, "aynı Aachen protokolü altında doğrudan
karşılaştırılabilirlik" iddiasını (§II-d, §V-B, Tablo III) geçersiz kılıyor:
Sueiras / Kang / Kass sayıları tam test setinde ölçülmüş, 84.54 ise dörtte
birinde. Sızıntı, sözlüğü yeniden kurarak düzelir; bu düzelmez, her şeyin
yeniden çalıştırılması gerekir. Görüntülerin tamamı diskte olduğu için tek
eksik doğru words.txt dosyası.

SENTETİK KAYITLAR — ETKİSİ SIFIR

words.txt'te 6,756 user-added-* ve 294 a01-999x kaydı var (toplam 7,050), bounding
box alanları sıfır veya sabit. Raporun ilk sürümü etkiyi "ihmal edilebilir"
diyordu; kontrol edildiğinde etki tam olarak sıfır: user öneki loader'da
form_id.startswith("user") ile eleniyor, a01-999x'in ise görüntü klasörü hiç
mevcut değil, cv2.imread None dönüp kayıt atlanıyor. Ne eğitime ne teste giriyor.
Bir repo hijyeni sorunu, sonuç geçerliliği sorunu değil.

SONUÇLAR

84.54%'ün bileşenleri, hepsi aynı model üzerinde:

| Bileşen                            |     WA |
|------------------------------------|--------|
| ham optik model                    |  74.97 |
| sızıntısız post-processing katkısı | +3.00 → 77.97 |
| test-set sızıntısı                 | +6.57 → 84.54 |

Writing-aware augmentation'ın ham çıktıdaki katkısı +1.82 pp, corrector sonrası
+1.42 pp (p = 0.0012). Elastic'in katkısı sıfır (p = 0.93), morfolojik bileşen
photo üstüne eklendiğinde zarar veriyor; yalnızca üçlü kombinasyon pozitif.
Makalenin merkezi iddiası — "augmentation, parametre maliyeti olmadan 6.48 pp
kazandırıyor" — desteklenmiyor. Ayakta kalan iddia: augmentation +1.8 pp,
lexicon+trigram +3.0 pp.

Ek olarak, f055c36 commit mesajı eski baseline'ın zaten "V3 Extended Trigram
(IAM + NLTK 235K)" kullandığını gösteriyor. Sözlük, makalede sunulduğu gibi
tamamen yeni bir bileşen değil; yeni olan, temiz kurulumda ölçülmüş katkısı.

DOĞRULAMALAR

Ham greedy iki bağımsız implementasyonda birebir aynı çıktı (74.9719 / 10.8308).
Checkpoint boyutu 114,954,520 bayt ÷ 4 = 28.74M parametre, makalenin CRNN-L'i
28.73M — mimari aynı. Commit f055c36, eski modelin ham greedy'sini 74.90 ve
trigram sonrasını 78.06 olarak bağımsız biçimde doğruluyor. Sızıntı, boyut
kontrollü deneyle izole edildi ve OOV tavanı words.txt üzerinden ikinci kez
hesaplandı. Tablo 2'nin tüm satırları per-sample CSV'lerden yeniden hesaplandı
ve JSON özetleriyle eşleşiyor. CSV'lerin satır hizası (0 ground-truth
uyuşmazlığı) doğrulandıktan sonra McNemar uygulandı.

KISITLAR

narrow run'ı CRNN-L'in tam reprodüksiyonu değil: aynı mimari ve benzer recipe
olmasına rağmen ham greedy'si 73.15, eski modelinki 74.90. Farklı seed ve
checkpoint seçimi. Bu nedenle "narrow 83.12 vs makale 78.06" karşılaştırması
temkinli kullanılmalı. Buna karşılık aynı olgu bağımsız bir kanıt da sunuyor:
augmentation'sız eski model ham greedy'de 74.90, full AugCRNN-T 74.97 — yani
script'ler arası varyans, augmentation'ın ölçülen katkısı kadar büyük.

Her konfigürasyon tek run, seed varyansı ölçülmedi. McNemar p değerleri örnekleme
gürültüsünü kapsıyor, eğitim gürültüsünü kapsamıyor.

Ham greedy karşılaştırmaları için McNemar yapılamadı; raw_greedy_eval2.py
per-sample tahminleri değil yalnızca toplamları kaydetti. Bu testler yalnızca
corrector sonrası CSV'ler üzerinden yapılabildi.

Temiz sözlük ölçümü yalnızca AugCRNN-T checkpoint'i için yapıldı. narrow'un temiz
sözlük altındaki değeri ölçülmedi; makalenin yeni ana tablosu bu hücre olmadan
yazılamaz.

77.97 BİR TAVAN DEĞİL, ALT SINIR

Bu rapordaki sızıntısız sayı (77.97) "modelin gerçek doğruluğu" olarak okunmamalı.
Kesik words.txt'in iki etkisi de sayıyı aşağı çekiyor, yukarı değil:

Birincisi, model verinin yarısından azıyla eğitildi. 498/747 form, yani 31,320
kelime; tam Aachen train'i ~55K kelime. 28.7M parametreli bir CRNN 31K kelimede
veri açlığı çekiyor ve bu, düşük ham greedy'nin (74.97) en olası açıklaması.
Görüntülerin tamamı (115,320 PNG) diskte olduğu için bu eksik ücretsiz kapanır.

İkincisi, temiz sözlük de kesik. Train-split sözlüğü 5,818 tip; tam train
split'iyle bu belirgin biçimde büyür — sızıntı olmadan, meşru şekilde. Sözlük
kapsamı arttıkça corrector'ın +3.00 pp'lik katkısı da artar.

Yani P0'daki yeniden eğitim "makale kurtulur mu" deneyi değil, "gerçek sayı ne"
deneyi. Sayının 77.97'nin üstünde çıkması beklenir; ne kadar üstünde olacağının
garantisi yok. Birkaç puan makul bir beklenti; 84.54'e dönmesi değil.

MAKALE NE OLARAK SAVUNULUR

Yeniden eğitim sonrası sayı 80-82 aralığında çıkarsa SOTA iddiası mümkün değil —
Kang 2021 84.09, Kass & Vats 84.60. Ama o iddia zaten hiçbir zaman gerçek
değildi. Dürüst konumlandırma şu: ön-eğitimsiz, sıfırdan eğitilmiş, 28.7M
parametreli kompakt bir CRNN, attention/transformer makinesi olmadan bu noktaya
geliyor. "En iyi" değil, "en basit makineyle şuraya kadar" makalesi. Bu haliyle
yayınlanabilir.

Ancak elde kalan en güçlü sonuç doğruluk sayısı değil. Bu çalışmanın ürettiği ve
literatürde nadiren nicelenmiş olan şey şu: IAM kelime tanımada sözlük
sızıntısının büyüklüğü. Boyut-kontrollü deneyle izole edilmiş +6.6 pp şişme,
%17.9'luk OOV tavanı, post-processing yığını değişiminin tek başına getirdiği
+6.2 pp, ve augmentation bileşenlerinin ayrı ayrı sıfır çıkması (p = 0.93) —
temiz bir negatif sonuç.

Bu çerçevede makalenin doğruluk sayısının 78 mi 84 mü olduğu önemsiz; hatta düşük
sayı argümanı güçlendiriyor: aynı model, aynı test seti, sadece sözlüğün nereden
kurulduğuna göre 72.89 ile 84.58 arasında herhangi bir sayı raporlanabiliyor. Bu
iddia için kimseyi geçmek gerekmiyor, yalnızca ölçümün doğru olması gerekiyor.

Kısacası savunulacak şey duruyor, savunulan şey değişiyor: "daha iyi bir tanıyıcı
yaptık" değil, "bu alandaki word accuracy sayıları göründüğü gibi değil, işte
niceliği".

ÖNERİLER

Öncelik sırasıyla:

P0 — Gerçek IAM words.txt (fki.tic.heia-fr.ch) indirilip veri seti yeniden
kurulmalı, narrow ve AugCRNN-T gerçek Aachen split'inde yeniden eğitilmeli.
Bu yapılmadan Tablo III'teki hiçbir dış karşılaştırma savunulabilir değil.

P1 — Sözlük Aachen train split'inden kurulmalı ve raporlanan her konfigürasyon
için temiz sözlükle ölçüm alınmalı, narrow dahil.

P2 — Konfigürasyon başına en az üç seed. Tek-seed sonuçlara dayanan ondalık
katkı iddialarından kaçınılmalı.

P3 — raw_greedy_eval2.py per-sample tahminleri yazsın ki ham greedy
karşılaştırmalarında da McNemar yapılabilsin.

Ayrıca: baseline ve önerilen sistem her zaman aynı script ve aynı post-processing
yığını ile ölçülmeli; augmentation ablation'ı ham greedy üzerinden raporlanmalı,
aksi halde corrector tavan etkisi farkları gizliyor.

Sızıntısız haliyle bugün savunulabilir sonuç: 73.15 → 77.97 (augmentation +1.82,
lexicon+trigram +3.00) — ve yukarıda açıklandığı gibi bu bir alt sınır.

Çerçeve kararı P0'dan sonraya bırakılmalı. Gerçek split'teki sayı 82+ çıkarsa iki
hikâye birleştirilebilir (kompakt sıfırdan-eğitilmiş tanıyıcı + sızıntı analizi);
79 civarında kalırsa sızıntı/metodoloji çalışması tek başına daha sağlam durur.

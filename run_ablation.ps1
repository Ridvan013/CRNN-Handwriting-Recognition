# Ablation egitimleri - PowerShell
#
# Kullanim (repo klasorunde):
#   .\run_ablation.ps1              -> narrow + full  (makalenin ana sayilari)
#   .\run_ablation.ps1 components   -> photo + elastic + morph
#   .\run_ablation.ps1 all          -> besi birden
#   .\run_ablation.ps1 elastic      -> tek mod
#
# Calistirma izni gerekirse:
#   powershell -ExecutionPolicy Bypass -File .\run_ablation.ps1 main

param(
    [string]$Set = "main",
    [int]$Epochs = 100,
    [int]$Batch = 128,
    [string]$Lr = "7e-4",
    [int]$Patience = 15
)

$ErrorActionPreference = "Continue"

# Python pipe'a yazarken ciktisini tamponlar; Tee-Object ile birlikte
# ilerleme ne konsolda ne log dosyasinda gorunur. Tamponlamayi kapatiyoruz.
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

switch ($Set) {
    "main"       { $modes = @("narrow", "full") }
    "components" { $modes = @("photo", "elastic", "morph") }
    "all"        { $modes = @("narrow", "full", "photo", "elastic", "morph") }
    default      { $modes = @($Set) }
}

New-Item -ItemType Directory -Force -Path logs | Out-Null

Write-Host "=============================================================="
Write-Host " Split dogrulamasi"
Write-Host "=============================================================="
python verify_aachen_splits.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "DOGRULAMA BASARISIZ - egitim baslatilmadi" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host ("Calistirilacak modlar: " + ($modes -join ", "))
Write-Host "epochs=$Epochs batch=$Batch lr=$Lr patience=$Patience"
Write-Host ""

$startAll = Get-Date
foreach ($m in $modes) {
    $dir = "Model_abl_$m"
    $log = "logs\abl_$m.log"
    Write-Host "=============================================================="
    Write-Host " --aug-mode $m   ->  $dir"
    Write-Host " log: $log"
    Write-Host (" baslangic: " + (Get-Date -Format "HH:mm:ss"))
    Write-Host "=============================================================="
    $t0 = Get-Date

    python cloud/v3_augmented_train.py --aug-mode $m --epochs $Epochs --batch $Batch --lr $Lr --patience $Patience --model-dir $dir 2>&1 | Tee-Object -FilePath $log -Encoding utf8

    $rc = $LASTEXITCODE
    $mins = [int]((Get-Date) - $t0).TotalMinutes
    if ($rc -ne 0) {
        Write-Host ">>> $m BASARISIZ (cikis $rc), sonraki moda geciliyor" -ForegroundColor Red
    } else {
        Write-Host ">>> $m tamamlandi - $mins dakika" -ForegroundColor Green
    }
    Write-Host ""
}

Write-Host "=============================================================="
Write-Host " Lexicon / trigram ablation"
Write-Host "=============================================================="
if (Test-Path "Model_abl_full\best_model_wa.pth") {
    python cloud/ablation_lexicon.py --model Model_abl_full/best_model_wa.pth --out results/ablation_lexicon.json 2>&1 | Tee-Object -FilePath "logs\ablation_lexicon.log"
} else {
    Write-Host "Model_abl_full\best_model_wa.pth yok - atlandi (once 'full' modunu egit)"
}

$totalMins = [int]((Get-Date) - $startAll).TotalMinutes
Write-Host ""
Write-Host "=============================================================="
Write-Host " OZET   (toplam $totalMins dakika)"
Write-Host "=============================================================="
python cloud/summarize_ablation.py

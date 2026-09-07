# Ablation egitimleri - PowerShell (Windows PowerShell 5.1 uyumlu)
#
#   .\run_ablation.ps1              -> narrow + full  (makalenin ana sayilari)
#   .\run_ablation.ps1 components   -> photo + elastic + morph
#   .\run_ablation.ps1 all          -> besi birden
#   .\run_ablation.ps1 elastic      -> tek mod
#
# Izin gerekirse:
#   powershell -ExecutionPolicy Bypass -File .\run_ablation.ps1 main

param(
    [string]$Set = "main",
    [int]$Epochs = 100,
    [int]$Batch = 128,
    [string]$Lr = "7e-4",
    [int]$Patience = 15,
    [int]$NumWorkers = 0,
    [int]$ElasticLegacy = 1,          # 1: orijinal genlik (~0.05 px, no-op)  0: alpha = RMS px
    [string]$ElasticAlpha = "2 5",    # ornek: -ElasticLegacy 0 -ElasticAlpha "1 3"
    [int]$GpuAug = 1
)
$ea = $ElasticAlpha -split " "

$ErrorActionPreference = "Continue"

# Python pipe'a yazarken ciktisini tamponlar; ilerleme gorunmez olur.
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
Write-Host "epochs=$Epochs batch=$Batch lr=$Lr patience=$Patience gpu-aug=$GpuAug"
Write-Host ("elastic: alpha " + $ElasticAlpha + "  " + $(if ($ElasticLegacy -eq 1) {"LEGACY genlik (~no-op)"} else {"RMS px (duzeltilmis)"}))
Write-Host ""

$startAll = Get-Date
$failed = @()

foreach ($m in $modes) {
    $dir = "Model_abl_$m"
    $log = Join-Path "logs" "abl_$m.log"
    Write-Host "=============================================================="
    Write-Host " --aug-mode $m   ->  $dir"
    Write-Host " log: $log"
    Write-Host (" baslangic: " + (Get-Date -Format "HH:mm:ss"))
    Write-Host "=============================================================="
    $t0 = Get-Date

    # Not: Windows PowerShell 5.1'de Tee-Object'in -Encoding parametresi YOK.
    # Satir satir hem ekrana hem UTF-8 log dosyasina yaziyoruz.
    if (Test-Path $log) { Remove-Item $log -Force }
    python cloud/v3_augmented_train.py --aug-mode $m --epochs $Epochs --batch $Batch --lr $Lr --patience $Patience --num-workers $NumWorkers --gpu-aug $GpuAug --elastic-legacy-amplitude $ElasticLegacy --elastic-alpha $ea[0] $ea[1] --model-dir $dir 2>&1 |
        ForEach-Object {
            $line = $_.ToString()
            Write-Host $line
            # Dosya baska bir surec tarafindan (izleyici/tail) kilitliyse kisa
            # bir yeniden deneme; yine olmazsa satiri atla, kosuyu bozma.
            $ok = $false
            for ($try = 0; $try -lt 5 -and -not $ok; $try++) {
                try { Add-Content -Path $log -Value $line -Encoding UTF8 -ErrorAction Stop; $ok = $true }
                catch { Start-Sleep -Milliseconds 100 }
            }
        }

    $mins = [int]((Get-Date) - $t0).TotalMinutes
    $ckpt = Join-Path $dir "best_model_wa.pth"
    $csv  = Join-Path $dir "test_results_analysis.csv"

    if (Test-Path $csv) {
        Write-Host ">>> $m tamamlandi - $mins dakika" -ForegroundColor Green
    } elseif (Test-Path $ckpt) {
        Write-Host ">>> $m YARIM KALDI ($mins dk): checkpoint var ama test ciktisi yok" -ForegroundColor Yellow
        $failed += $m
    } else {
        Write-Host ">>> $m BASARISIZ ($mins dk): hic checkpoint olusmadi. Log: $log" -ForegroundColor Red
        $failed += $m
    }
    Write-Host ""
}

Write-Host "=============================================================="
Write-Host " Lexicon / trigram ablation"
Write-Host "=============================================================="
if (Test-Path "Model_abl_full\best_model_wa.pth") {
    if (Test-Path "logs\ablation_lexicon.log") { Remove-Item "logs\ablation_lexicon.log" -Force }
    python cloud/ablation_lexicon.py --model Model_abl_full/best_model_wa.pth --out results/ablation_lexicon.json 2>&1 |
        ForEach-Object {
            $line = $_.ToString()
            Write-Host $line
            Add-Content -Path "logs\ablation_lexicon.log" -Value $line -Encoding UTF8
        }
} else {
    Write-Host "Model_abl_full\best_model_wa.pth yok - atlandi"
}

$totalMins = [int]((Get-Date) - $startAll).TotalMinutes
Write-Host ""
Write-Host "=============================================================="
Write-Host " OZET   (toplam $totalMins dakika)"
Write-Host "=============================================================="
python cloud/summarize_ablation.py

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host ("TAMAMLANMAYAN MODLAR: " + ($failed -join ", ")) -ForegroundColor Red
    exit 1
}

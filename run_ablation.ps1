# Augmentation ablation trainings - PowerShell (Windows PowerShell 5.1 compatible)
#
#   .\run_ablation.ps1              -> narrow + full  (the two endpoint configurations)
#   .\run_ablation.ps1 components   -> photo + elastic + morph
#   .\run_ablation.ps1 all          -> all six, including the zero-augmentation anchor
#   .\run_ablation.ps1 elastic      -> a single mode
#
# If the execution policy blocks the script:
#   powershell -ExecutionPolicy Bypass -File .\run_ablation.ps1 main

param(
    [string]$Set = "main",
    [int]$Epochs = 100,
    [int]$Batch = 128,
    [string]$Lr = "7e-4",
    [int]$Patience = 15,
    [int]$NumWorkers = 0,
    [int]$ElasticLegacy = 0,          # 0 (paper): alpha = RMS px   1: earlier no-op amplitude
    [string]$ElasticAlpha = "1 3",    # paper setting; the earlier code used "2 5" with -ElasticLegacy 1
    [int]$GpuAug = 1
)
$ea = $ElasticAlpha -split " "

$ErrorActionPreference = "Continue"

# Python buffers its output when writing to a pipe; progress would not show.
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

switch ($Set) {
    "main"       { $modes = @("narrow", "full") }
    "components" { $modes = @("photo", "elastic", "morph") }
    "all"        { $modes = @("none", "narrow", "full", "photo", "elastic", "morph") }
    default      { $modes = @($Set) }
}

New-Item -ItemType Directory -Force -Path logs | Out-Null

Write-Host "=============================================================="
Write-Host " Split verification"
Write-Host "=============================================================="
python verify_aachen_splits.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "VERIFICATION FAILED - training not started" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host ("Modes to run: " + ($modes -join ", "))
Write-Host "epochs=$Epochs batch=$Batch lr=$Lr patience=$Patience gpu-aug=$GpuAug"
Write-Host ("elastic: alpha " + $ElasticAlpha + "  " + $(if ($ElasticLegacy -eq 1) {"LEGACY amplitude (~no-op)"} else {"RMS px (corrected)"}))
Write-Host ""

$startAll = Get-Date
$failed = @()

foreach ($m in $modes) {
    $dir = "Model_abl_$m"
    $log = Join-Path "logs" "abl_$m.log"
    Write-Host "=============================================================="
    Write-Host " --aug-mode $m   ->  $dir"
    Write-Host " log: $log"
    Write-Host (" start: " + (Get-Date -Format "HH:mm:ss"))
    Write-Host "=============================================================="
    $t0 = Get-Date

    # Note: in Windows PowerShell 5.1 Tee-Object has no -Encoding parameter,
    # so every line is written both to the screen and to a UTF-8 log file.
    if (Test-Path $log) { Remove-Item $log -Force }
    python cloud/v3_augmented_train.py --aug-mode $m --epochs $Epochs --batch $Batch --lr $Lr --patience $Patience --num-workers $NumWorkers --gpu-aug $GpuAug --elastic-legacy-amplitude $ElasticLegacy --elastic-alpha $ea[0] $ea[1] --model-dir $dir 2>&1 |
        ForEach-Object {
            $line = $_.ToString()
            Write-Host $line
            # If another process (a viewer, tail) holds the file, retry briefly;
            # if it still fails, drop the line rather than break the run.
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
        Write-Host ">>> $m finished - $mins minutes" -ForegroundColor Green
    } elseif (Test-Path $ckpt) {
        Write-Host ">>> $m INCOMPLETE ($mins min): checkpoint present but no test output" -ForegroundColor Yellow
        $failed += $m
    } else {
        Write-Host ">>> $m FAILED ($mins min): no checkpoint was written. Log: $log" -ForegroundColor Red
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
    Write-Host "Model_abl_full\best_model_wa.pth missing - skipped"
}

$totalMins = [int]((Get-Date) - $startAll).TotalMinutes
Write-Host ""
Write-Host "=============================================================="
Write-Host " SUMMARY   (total $totalMins minutes)"
Write-Host "=============================================================="
python cloud/summarize_ablation.py

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host ("MODES NOT COMPLETED: " + ($failed -join ", ")) -ForegroundColor Red
    exit 1
}

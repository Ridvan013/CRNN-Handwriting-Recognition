#!/usr/bin/env python3
"""Calisan ablation egitiminin durumunu gosterir (log tamponlu olsa bile)."""
import os, subprocess, time, glob, json
from datetime import datetime

MODES = ["narrow", "full", "photo", "elastic", "morph"]
print("=" * 62)
print(f"  ABLATION DURUMU   {datetime.now():%H:%M:%S}")
print("=" * 62)

try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"  GPU: {out}")
except Exception:
    pass

# Egitim surecini komut satirindan tespit et.
#  - tasklist yalnizca "python.exe" der; bu betigin KENDISI de python.exe
#    oldugu icin onu saymak yanilticiydi.
#  - wmic Windows 11 24H2+ surumlerinde KALDIRILDI, bos donuyordu ve
#    "EGITIM CALISMIYOR" diye yanlis alarm veriyordu.
# Bu yuzden once psutil, olmazsa PowerShell Get-CimInstance kullaniyoruz.
MARKERS = ("v3_augmented_train", "ablation_lexicon")


def _training_procs():
    try:
        import psutil
        out = []
        for pr in psutil.process_iter(["name", "cmdline"]):
            try:
                cmd = " ".join(pr.info["cmdline"] or [])
            except Exception:
                continue
            if any(m in cmd for m in MARKERS):
                out.append(pr.pid)
        return out, "psutil"
    except ImportError:
        pass
    try:
        ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              "Where-Object { $_.CommandLine -match "
              "'v3_augmented_train|ablation_lexicon' } | "
              "Select-Object -ExpandProperty ProcessId")
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=30).stdout
        return [ln.strip() for ln in r.splitlines() if ln.strip().isdigit()], "powershell"
    except Exception as exc:
        return None, f"tespit edilemedi ({exc})"


pids, how = _training_procs()
if pids is None:
    print(f"  Egitim sureci: BILINMIYOR  ({how})")
elif pids:
    print(f"  Egitim sureci: {len(pids)} adet {tuple(pids)}  -> CALISIYOR   [{how}]")
else:
    print(f"  Egitim sureci: YOK  -> EGITIM CALISMIYOR   [{how}]")
    print("     (bittiyse asagida 'BITTI' gorursunuz; gorunmuyorsa kosu"
          " yarida kesilmis demektir)")
print()

for m in MODES:
    d = next((c for c in (f"Model_abl_{m}", f"abl_{m}") if os.path.isdir(c)), None)
    if not d:
        continue
    ck = os.path.join(d, "best_model_wa.pth")
    hist = os.path.join(d, "training_history.json")
    csvp = os.path.join(d, "test_results_analysis.csv")
    line = f"  {m:<8}"
    if os.path.exists(csvp):
        n = sum(1 for _ in open(csvp, encoding="utf-8")) - 1
        line += f"BITTI   test ciktisi {n:,} satir"
    elif os.path.exists(hist):
        h = json.load(open(hist))
        ep = len(h.get("train_loss", []))
        best = max(h.get("val_wa", [0])) * 100
        line += f"egitim bitti, test suruyor  ({ep} epoch, en iyi val WA {best:.2f}%)"
    elif os.path.exists(ck):
        age = (time.time() - os.path.getmtime(ck)) / 60
        line += (f"egitiliyor  son iyilesme {age:.0f} dk once "
                 f"({datetime.fromtimestamp(os.path.getmtime(ck)):%H:%M})")
    else:
        line += "basladi, henuz checkpoint yok (veri yukleniyor olabilir)"
    print(line)

print()
print("  Not: en iyi checkpoint yalnizca DOGRULUK ARTTIGINDA yazilir.")
print("       Uzun sure guncellenmemesi normaldir (plato / erken durdurma yaklasiyor).")

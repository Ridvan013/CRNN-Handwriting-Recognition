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

running = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"],
                         capture_output=True, text=True).stdout
n_py = running.count("python.exe")
print(f"  Calisan python sureci: {n_py}"
      + ("  -> egitim suruyor" if n_py else "  -> egitim CALISMIYOR"))
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

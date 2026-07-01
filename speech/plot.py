"""Generate benchmark charts (PNG) from the speech CSVs. Run: python plot.py"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")
PURPLE = "#6D28D9"


def read_csv(name):
    with open(os.path.join(HERE, name)) as f:
        return list(csv.DictReader(f))


# STT — accuracy (WER) vs speed (RTFx, end-to-end)
stt = read_csv("stt.csv")
fig, ax = plt.subplots(figsize=(7, 4.5))
for row in stt:
    x, y = float(row["rtfx_median_end_to_end"]), float(row["wer_pct"])
    ax.scatter(x, y, s=110, color=PURPLE, zorder=3)
    ax.annotate(row["model"], (x, y), xytext=(7, 6), textcoords="offset points", fontsize=9)
ax.set_xlabel("RTFx (end-to-end via API) — higher is faster")
ax.set_ylabel("WER % — lower is better")
ax.set_title("Speech-to-text: accuracy vs. speed")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "stt-wer-vs-rtfx.png"), dpi=150)

# TTS — time to first audio (end-to-end)
tts = read_csv("tts.csv")
models = [r["model"] for r in tts]
ttfa = [float(r["ttfa_ms_end_to_end"]) for r in tts]
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(models, ttfa, color=PURPLE, width=0.5, zorder=3)
ax.bar_label(bars, fmt="%.0f ms", padding=3, fontsize=9)
ax.set_ylabel("TTFA (ms, end-to-end via API) — lower is better")
ax.set_title("Text-to-speech: time to first audio")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "tts-ttfa.png"), dpi=150)

print("wrote charts to", ASSETS)

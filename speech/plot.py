"""Regenerate the benchmark charts (assets/*.png) from the CSV data."""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")
PURPLE = "#6D28D9"
GREY = "#9CA3AF"


def read_csv(name):
    with open(os.path.join(HERE, name)) as f:
        return list(csv.DictReader(f))


# STT: WER vs RTFx (Open ASR Leaderboard, A100), EcoHash models highlighted.
fig, ax = plt.subplots(figsize=(6.2, 5))
for r in read_csv("stt.csv"):
    try:
        wer, rtfx = float(r["wer_pct"]), float(r["rtfx_a100_bs64"])
    except ValueError:
        continue
    eco = r["on_ecohash"].strip().lower() == "yes"
    ax.scatter(wer, rtfx, s=95 if eco else 42, c=PURPLE if eco else GREY,
               edgecolors="white", linewidths=0.6, zorder=3 if eco else 2)
    if eco:
        ax.annotate(r["model"].split("/")[-1], (wer, rtfx), fontsize=7.5,
                    fontweight="bold", xytext=(5, 4), textcoords="offset points")
ax.set_yscale("log")
ax.set_xlabel("WER %  (lower is better)")
ax.set_ylabel("RTFx on A100, batch 64  (higher is faster, log)")
ax.set_title("Open speech-to-text: accuracy vs speed\nHF Open ASR Leaderboard (A100) — purple = served on EcoHash")
ax.grid(True, which="both", alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "stt-wer-vs-rtfx.png"), dpi=140)

# TTS: time to first audio (mixed measured/claimed), EcoHash models highlighted.
pts = [(r["model"], float(r["ttfa_ms"]), r["on_ecohash"].strip().lower() == "yes")
       for r in read_csv("tts.csv") if r.get("ttfa_ms")]
pts.sort(key=lambda x: x[1], reverse=True)
fig, ax = plt.subplots(figsize=(6.2, 5))
ax.barh([p[0] for p in pts], [p[1] for p in pts],
        color=[PURPLE if p[2] else GREY for p in pts])
ax.set_xlabel("TTFA ms  (lower is better; mixed measured / vendor-claimed)")
ax.set_title("Open text-to-speech: time to first audio\npurple = served on EcoHash")
ax.grid(True, axis="x", alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "tts-ttfa.png"), dpi=140)

print("charts written to", os.path.normpath(ASSETS))

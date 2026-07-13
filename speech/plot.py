"""Regenerate all benchmark charts (assets/*.png) from the CSV data."""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")
PURPLE = "#6D28D9"
GREY = "#9CA3AF"


def read_csv(relpath):
    with open(os.path.join(ROOT, relpath)) as f:
        return list(csv.DictReader(f))


def barh(labels, values, eco_flags, title, xlabel, fname, fmt="{:.0f}"):
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.barh(labels, values, color=[PURPLE if e else GREY for e in eco_flags])
    for i, v in enumerate(values):
        ax.annotate(fmt.format(v), (v, i), fontsize=8, va="center",
                    xytext=(4, 0), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, fname), dpi=140)
    plt.close(fig)


def scatter(points, xlabel, ylabel, title, fname, ylog=False):
    # points: list of (label, x, y, eco, annotate)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for (label, x, y, eco, ann) in points:
        ax.scatter(x, y, s=120 if eco else 46, c=PURPLE if eco else GREY,
                   edgecolors="white", linewidths=0.7, zorder=3 if eco else 2)
        if ann:
            ax.annotate(label, (x, y), fontsize=8, fontweight="bold" if eco else "normal",
                        color=PURPLE if eco else "#555", xytext=(6, 4), textcoords="offset points")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, fname), dpi=140)
    plt.close(fig)


# STT: WER vs RTFx from the Open ASR Leaderboard (A100); EcoHash-served highlighted.
fig, ax = plt.subplots(figsize=(6.2, 5))
for r in read_csv("speech/stt.csv"):
    if "Leaderboard" not in r["source"]:
        continue
    try:
        wer, rtfx = float(r["wer_pct"]), float(r["rtfx"])
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
plt.close(fig)

# TTS: time to first audio across open models; EcoHash-served highlighted.
pts = [(r["model"], float(r["ttfa_ms"]), r["on_ecohash"].strip().lower() == "yes")
       for r in read_csv("speech/tts.csv") if r["source"] == "Open model" and r.get("ttfa_ms")]
pts.sort(key=lambda x: x[1], reverse=True)
barh([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts],
     "Open text-to-speech: time to first audio\npurple = served on EcoHash",
     "TTFA ms  (lower is better; mixed measured / vendor-claimed)", "tts-ttfa.png")

# LLM latency scatter: TTFT vs TPOT for Llama-3.1-8B across providers. Lower-left is better.
prov = read_csv("llm/llama8b-providers.csv")
lat = []
for r in prov:
    if not r.get("ttft_ms") or not r.get("tok_s"):
        continue
    eco = r["is_ecohash"].strip().lower() == "yes"
    tpot = 1000.0 / float(r["tok_s"])
    ann = eco or r["provider"] in ("Groq",)
    lat.append((r["provider"], float(r["ttft_ms"]), tpot, eco, ann))
scatter(lat, "TTFT ms  (lower is better)", "TPOT ms/token  (lower is better, log)",
        "Llama-3.1-8B latency: TTFT vs per-token\npurple = EcoHash (RTX PRO 6000), grey = peers (Artificial Analysis)",
        "llm-ttft-tpot.png", ylog=True)

# LLM value scatter: blended price vs output speed. EcoHash purple.
val = []
for r in prov:
    if not r.get("price_blended") or not r.get("tok_s"):
        continue
    eco = r["is_ecohash"].strip().lower() == "yes"
    ann = eco or r["provider"] in ("Groq",)
    val.append((r["provider"], float(r["price_blended"]), float(r["tok_s"]), eco, ann))
scatter(val, "Blended price $/1M tokens  (lower is better)", "Output tokens/sec  (higher is better)",
        "Llama-3.1-8B price vs speed\npurple = EcoHash, grey = peers (Artificial Analysis)",
        "llm-price-speed.png")

# Image has no peer comparison data, so it is shown as a table plus sample images in the
# README rather than an EcoHash-only chart.

# Small variants (560px wide) for embedding in blog posts, where plain Markdown
# image syntax renders at natural size.
from PIL import Image  # noqa: E402

for name in ["llm-ttft-tpot", "llm-price-speed", "stt-wer-vs-rtfx", "tts-ttfa"]:
    src = os.path.join(ASSETS, f"{name}.png")
    im = Image.open(src)
    w = 560
    h = round(im.height * w / im.width)
    im.resize((w, h), Image.LANCZOS).save(os.path.join(ASSETS, f"{name}-sm.png"))

print("charts written to", os.path.normpath(ASSETS))

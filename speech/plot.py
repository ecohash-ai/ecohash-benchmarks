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

# Same open model, different providers. Two charts each: TTFT vs TPOT, and blended
# price vs output speed. Peer numbers come from third parties, not measured here.
PROVIDER_SETS = [
    # csv, model label, peer source, filename suffix, peers worth labelling
    ("llm/llama8b-providers.csv", "Llama-3.1-8B", "Artificial Analysis", "", ("Groq",)),
    ("llm/qwen38-providers.csv", "Qwen3.8-27B", "OpenRouter", "-qwen38", ("Venice",)),
]

for relpath, model, peer_source, suffix, label_peers in PROVIDER_SETS:
    lat, val = [], []
    for r in read_csv(relpath):
        eco = r["is_ecohash"].strip().lower() == "yes"
        ann = eco or r["provider"] in label_peers
        if r.get("ttft_ms") and r.get("tok_s"):
            lat.append((r["provider"], float(r["ttft_ms"]),
                        1000.0 / float(r["tok_s"]), eco, ann))
        if r.get("price_blended") and r.get("tok_s"):
            val.append((r["provider"], float(r["price_blended"]),
                        float(r["tok_s"]), eco, ann))
    scatter(lat, "TTFT ms  (lower is better)", "TPOT ms/token  (lower is better, log)",
            f"{model} latency: TTFT vs per-token\n"
            f"purple = EcoHash (RTX PRO 6000), grey = peers ({peer_source})",
            f"llm-ttft-tpot{suffix}.png", ylog=True)
    scatter(val, "Blended price $/1M tokens  (lower is better)",
            "Output tokens/sec  (higher is better)",
            f"{model} price vs speed\npurple = EcoHash, grey = peers ({peer_source})",
            f"llm-price-speed{suffix}.png")

# Blended price alone, as a bar. The scatter above shows price against speed; this one
# exists because the price gap is the part that survives being read on a phone.
prov38 = read_csv("llm/qwen38-providers.csv")
prov38.sort(key=lambda r: float(r["price_blended"]), reverse=True)
barh([r["provider"] for r in prov38],
     [float(r["price_blended"]) for r in prov38],
     [r["is_ecohash"].strip().lower() == "yes" for r in prov38],
     "Qwen3.8-27B blended price\npurple = EcoHash, grey = peers (OpenRouter)",
     "Blended $/1M tokens, (3x input + 1x output) / 4  (lower is better)",
     "qwen38-price.png", fmt="${:.3f}")

# The provider comparison as a table image. The blog's editor is TipTap without the table
# extension, so a real <table> is dropped on paste; an image is the only way that grid
# survives into a post. Time to a finished answer is TTFT + tokens / throughput.
prov38 = read_csv("llm/qwen38-providers.csv")
order = sorted(prov38, key=lambda r: (r["is_ecohash"].strip().lower() != "yes", float(r["ttft_ms"])))
header = ["Provider", "First token", "Tokens/sec", "500-token\nanswer", "2,000-token\nanswer", "Price in / out\nper 1M"]
body = []
for r in order:
    ttft, tps = float(r["ttft_ms"]) / 1000.0, float(r["tok_s"])
    # \$ escaped: an unescaped $ opens matplotlib mathtext and the sign vanishes.
    body.append([r["provider"], f"{float(r['ttft_ms']):.0f} ms", f"{tps:.0f}",
                 f"{ttft + 500/tps:.1f} s", f"{ttft + 2000/tps:.1f} s",
                 rf"\${float(r['price_in']):.2f} / \${float(r['price_out']):.2f}"])

fig, ax = plt.subplots(figsize=(8.2, 0.52 * (len(body) + 1.8)))
ax.axis("off")
tbl = ax.table(cellText=body, colLabels=header, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.9)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#E5E7EB")
    if row == 0:
        cell.set_facecolor("#F3F4F6")
        cell.set_text_props(weight="bold", color="#111827")
    elif order[row - 1]["is_ecohash"].strip().lower() == "yes":
        cell.set_facecolor("#EDE9FE")
        cell.set_text_props(weight="bold", color=PURPLE)
    if col == 0:
        cell.set_text_props(ha="left")
        cell.PAD = 0.04
ax.set_title("Qwen3.8-27B across providers\nEcoHash measured end to end; peers as published by OpenRouter, 2026-08",
             fontsize=10.5, pad=14)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "qwen38-comparison.png"), dpi=140, bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# MTP: how the draft depth K trades off against acceptance. Grouped bars, one group per
# load, because the three loads separate more than the K values do.
kv = read_csv("llm/qwen38-mtp-kvalues.csv")
LOADS = ["math", "code", "chat"]
CONFIGS = [("baseline", "0"), ("mtp", "1"), ("mtp", "2"), ("mtp", "3")]
LABELS = ["no MTP", "K=1", "K=2", "K=3"]
SHADES = ["#9CA3AF", "#C4B5FD", "#A78BFA", PURPLE]

fig, ax = plt.subplots(figsize=(6.2, 5))
width = 0.2
for i, ((cfg, k), label, shade) in enumerate(zip(CONFIGS, LABELS, SHADES)):
    vals = [float(r["single_stream_tok_s"]) for load in LOADS for r in kv
            if r["config"] == cfg and r["k"] == k and r["load"] == load]
    xs = [j + (i - 1.5) * width for j in range(len(LOADS))]
    ax.bar(xs, vals, width, label=label, color=shade)
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:.0f}", (x, v), fontsize=7, ha="center",
                    xytext=(0, 2), textcoords="offset points")
ax.set_xticks(range(len(LOADS)))
ax.set_xticklabels(LOADS)
ax.set_ylabel("Single-stream tokens/sec  (higher is better)")
ax.set_title("Qwen3.8-27B: MTP draft depth, one RTX PRO 6000\nconcurrency 1, cluster-direct")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "qwen38-mtp-kvalues.png"), dpi=140)
plt.close(fig)

# Acceptance falls off with draft position, and it falls off much faster on chat than on
# math. That gap is why the speedup is not the same on every workload.
acc = [r for r in read_csv("llm/qwen38-mtp-acceptance.csv") if r["k"] == "3"]
fig, ax = plt.subplots(figsize=(6.2, 5))
positions = ["pos_1", "pos_2", "pos_3"]
for i, r in enumerate(acc):
    vals = [float(r[p]) for p in positions]
    xs = [j + (i - 1) * 0.25 for j in range(3)]
    ax.bar(xs, vals, 0.25, label=r["load"],
           color=[PURPLE, "#A78BFA", "#C4B5FD"][i])
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:.2f}", (x, v), fontsize=7, ha="center",
                    xytext=(0, 2), textcoords="offset points")
ax.set_xticks(range(3))
ax.set_xticklabels(["1st draft token", "2nd", "3rd"])
ax.set_ylabel("Acceptance rate  (higher is better)")
ax.set_ylim(0, 1)
ax.set_title("Qwen3.8-27B: draft acceptance by position, K=3\nacceptance depends on the content, not on load")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "qwen38-mtp-acceptance.png"), dpi=140)
plt.close(fig)

# Concurrency sweeps. Two charts off the same CSV: per-token latency against the SLO
# line, and aggregate output. Both run the full sweep, including the levels past the
# point where the SLO is already broken.
conc = read_csv("llm/qwen38-mtp-concurrency.csv")
by_load = {load: sorted([r for r in conc if r["load"] == load],
                        key=lambda r: int(r["concurrency"])) for load in ("math", "chat")}
STYLE = {"math": "-", "chat": "--"}

fig, ax = plt.subplots(figsize=(6.2, 5))
for load, rows in by_load.items():
    xs = [int(r["concurrency"]) for r in rows]
    ax.plot(xs, [float(r["baseline_tpot_p95_ms"]) for r in rows], STYLE[load],
            color=GREY, marker="o", ms=3.5, label=f"no MTP, {load}")
    ax.plot(xs, [float(r["mtp_tpot_p95_ms"]) for r in rows], STYLE[load],
            color=PURPLE, marker="o", ms=3.5, label=f"MTP K=3, {load}")
ax.axhline(30, color="#C00000", lw=1, ls=":")
ax.annotate("SLO: p95 TPOT 30 ms/tok", (1.2, 31), fontsize=7.5, color="#C00000")
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Concurrent requests")
ax.set_ylabel("p95 TPOT ms/token  (lower is better)")
ax.set_title("Qwen3.8-27B: per-token latency under load\none RTX PRO 6000, cluster-direct")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "qwen38-mtp-tpot.png"), dpi=140)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 5))
for load, rows in by_load.items():
    xs = [int(r["concurrency"]) for r in rows]
    ax.plot(xs, [float(r["baseline_out_tok_s"]) for r in rows], STYLE[load],
            color=GREY, marker="o", ms=3.5, label=f"no MTP, {load}")
    ax.plot(xs, [float(r["mtp_out_tok_s"]) for r in rows], STYLE[load],
            color=PURPLE, marker="o", ms=3.5, label=f"MTP K=3, {load}")
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Concurrent requests")
ax.set_ylabel("Aggregate output tokens/sec  (higher is better)")
ax.set_title("Qwen3.8-27B: aggregate output under load\nMTP leads up to the SLO ceiling; the lines meet past it")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "qwen38-mtp-throughput.png"), dpi=140)
plt.close(fig)

# Image has no peer comparison data, so it is shown as a table plus sample images in the
# README rather than an EcoHash-only chart.

# Small variants (560px wide) for embedding in blog posts, where plain Markdown
# image syntax renders at natural size.
from PIL import Image  # noqa: E402

for name in ["llm-ttft-tpot", "llm-price-speed",
             "llm-ttft-tpot-qwen38", "llm-price-speed-qwen38", "qwen38-price", "qwen38-comparison",
             "qwen38-mtp-kvalues", "qwen38-mtp-acceptance",
             "qwen38-mtp-tpot", "qwen38-mtp-throughput",
             "stt-wer-vs-rtfx", "tts-ttfa"]:
    src = os.path.join(ASSETS, f"{name}.png")
    im = Image.open(src)
    w = 560
    h = round(im.height * w / im.width)
    im.resize((w, h), Image.LANCZOS).save(os.path.join(ASSETS, f"{name}-sm.png"))

print("charts written to", os.path.normpath(ASSETS))


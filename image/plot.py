"""Regenerate image-generation benchmark charts (assets/image-*.png) from image.csv."""

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

rows = []
with open(os.path.join(HERE, "image.csv")) as f:
    for r in csv.DictReader(f):
        rows.append(r)

# Time per step, all four models measured end to end on one RTX PRO 6000.
# Highlight qwen-image-2.1: fastest per-step time in this set, day-one benchmark.
DISPLAY_NAME = {"qwen-image-2-1": "qwen-image-2.1"}

rows.sort(key=lambda r: float(r["per_step_s"]))
labels = [DISPLAY_NAME.get(r["model"], r["model"]) for r in rows]
values = [float(r["per_step_s"]) for r in rows]
highlight = [r["model"] == "qwen-image-2-1" for r in rows]

fig, ax = plt.subplots(figsize=(6.2, 4))
bars = ax.barh(labels, values, color=[PURPLE if h else GREY for h in highlight])
for i, v in enumerate(values):
    ax.annotate(f"{v:.2f}s", (v, i), fontsize=9, va="center",
                xytext=(5, 0), textcoords="offset points",
                fontweight="bold" if highlight[i] else "normal",
                color=PURPLE if highlight[i] else "#374151")
ax.set_xlabel("Time per step, seconds (lower is faster)")
ax.set_title("Image generation: time per step on RTX PRO 6000\npurple = Qwen-Image-2.1, measured 2026-09-21")
ax.set_xlim(0, max(values) * 1.35)
ax.grid(True, axis="x", alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "image-per-step.png"), dpi=140)
plt.close(fig)

# Small variant (560px wide) for embedding in blog posts / X, matching the
# convention in speech/plot.py.
from PIL import Image  # noqa: E402

src = os.path.join(ASSETS, "image-per-step.png")
im = Image.open(src)
w = 560
h = round(im.height * w / im.width)
im.resize((w, h), Image.LANCZOS).save(os.path.join(ASSETS, "image-per-step-sm.png"))

print("chart written to", os.path.normpath(ASSETS))

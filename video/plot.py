"""Regenerate the video-generation charts (assets/video-*.png, assets/h3-*.png).

Reads the CSVs next to this file. Same palette and sizing as speech/plot.py so the
charts sit next to the others without looking imported from somewhere else.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")
PURPLE = "#6D28D9"
GREY = "#9CA3AF"
LIGHT = "#DDD6FE"


def read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, fname):
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, fname), dpi=140)
    plt.close(fig)
    print("  wrote assets/" + fname)


def before_after():
    rows = read("video.csv")
    labels, before, after, speed = [], [], [], []
    for r in rows:
        labels.append("%s\n%s" % (r["model"], r["variant"]))
        before.append(float(r["before_s"]))
        after.append(float(r["after_s"]))
        speed.append(float(r["speedup"]))
    y = np.arange(len(labels))
    h = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 5))
    ax.barh(y + h / 2, before, h, color=GREY, label="before")
    ax.barh(y - h / 2, after, h, color=PURPLE, label="after")
    for i, (b, a, s) in enumerate(zip(before, after, speed)):
        ax.annotate("%.1f s" % b, (b, i + h / 2), fontsize=7.5, va="center",
                    xytext=(4, 0), textcoords="offset points", color="#555")
        ax.annotate("%.1f s   %.2fx" % (a, s), (a, i - h / 2), fontsize=7.5,
                    va="center", xytext=(4, 0), textcoords="offset points",
                    color=PURPLE, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(5, 3000)
    ax.set_xlabel("seconds per clip, log scale")
    ax.set_title("Video generation on one RTX PRO 6000\nbefore and after optimisation")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    save(fig, "video-before-after.png")


def techniques():
    rows = read("video-techniques.csv")
    order = ["minimax-h3", "wan22-t2v-a14b", "wan22-s2v-14b"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.2))
    for ax, model in zip(axes, order):
        rs = [r for r in rows if r["model"] == model]
        labels = [r["config"] for r in rs]
        vals = [float(r["latency_s"]) for r in rs]
        ship = [r["shipped"] == "yes" for r in rs]
        y = np.arange(len(labels))
        ax.barh(y, vals, color=[PURPLE if s else GREY for s in ship])
        for i, (v, s) in enumerate(zip(vals, ship)):
            ax.annotate("%.1f" % v, (v, i), fontsize=7.5, va="center",
                        xytext=(3, 0), textcoords="offset points",
                        color=PURPLE if s else "#555")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.28)
        ax.set_xlabel("seconds")
        ax.set_title(model, fontsize=10)
        ax.grid(True, axis="x", alpha=0.2)
    fig.suptitle("What each step bought, purple = shipped to production", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(ASSETS, "video-techniques.png"), dpi=140)
    plt.close(fig)
    print("  wrote assets/video-techniques.png")


def h3_stages():
    """Grouped, not stacked.

    Stacking put the three small stages on top of each other and their labels
    collided. It also invited reading the stack height as the wall clock, which
    it is not: the stage medians come from different prompts, so they sum to
    within a few tenths of the measured total rather than exactly.
    """
    rows = [r for r in read("h3-stage-breakdown.csv") if r["stage"] != "total"]
    stages = [r["stage"] for r in rows]
    before = [float(r["before_s"]) for r in rows]
    after = [float(r["after_s"]) for r in rows]
    y = np.arange(len(stages))
    h = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 5))
    ax.barh(y + h / 2, before, h, color=GREY, label="before: 20 steps, int8")
    ax.barh(y - h / 2, after, h, color=PURPLE, label="after: 8 steps, fp8")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.annotate("%.1f s" % b, (b, i + h / 2), fontsize=8, va="center",
                    xytext=(4, 0), textcoords="offset points", color="#555")
        ax.annotate("%.1f s" % a, (a, i - h / 2), fontsize=8, va="center",
                    xytext=(4, 0), textcoords="offset points", color=PURPLE,
                    fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(stages, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, 400)
    ax.set_xlabel("seconds per 3.04 s clip, log scale")
    ax.set_title("MiniMax H3: where the time goes\n"
                 "wall clock 174.8 s to 40.8 s; only sampling moved")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    save(fig, "h3-stage-breakdown.png")


def h3_length():
    rows = read("h3-length-scaling.csv")
    frames = [int(r["frames"]) for r in rows]
    cost = [float(r["gpu_s_per_video_s"]) for r in rows]
    secs = [float(r["clip_seconds"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.plot(secs, cost, marker="o", color=PURPLE, linewidth=2, markersize=8)
    for s, c, f in zip(secs, cost, frames):
        ax.annotate("%d frames\n%.1f" % (f, c), (s, c), fontsize=8,
                    xytext=(8, -4), textcoords="offset points", color="#444")
    ax.set_xlabel("clip length, seconds")
    ax.set_ylabel("GPU seconds per second of video")
    ax.set_ylim(0, 21)
    ax.set_xlim(2, 10)
    ax.set_title("MiniMax H3: longer clips cost more per second\n"
                 "attention is quadratic in sequence length")
    ax.grid(True, alpha=0.2)
    save(fig, "h3-length-scaling.png")


if __name__ == "__main__":
    before_after()
    techniques()
    h3_stages()
    h3_length()

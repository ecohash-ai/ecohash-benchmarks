#!/usr/bin/env python3
"""
benchmark.py — measure quality and speed of the speech models served on EcoHash.

What it measures:
  STT (speech-to-text): WER (word error rate on labeled audio) + RTFx (audio duration / latency)
  TTS (text-to-speech): TTFA (time to first audio) + RTF (synthesis time / audio duration)

How to run:
  1) pip install openai jiwer datasets soundfile numpy requests
  2) export ECOHASH_API_KEY=eco_...
     python benchmark.py
  Defaults to https://api.ecohash.com/v1, auto-discovers the platform's active speech
  models, uses a small LibriSpeech slice for STT (WER/RTFx) and built-in sentences for TTS.

Notes:
  - Measured end-to-end through the API, so results include network + queueing (not
    pure-GPU throughput). Warmup runs + median are used to suppress cold starts.
  - WER uses a simple English normalizer (lowercase, strip punctuation), not the official
    Open ASR normalizer, so it is for cross-model comparison only.
"""
import argparse, io, os, re, statistics, sys, time, wave, json
import requests

DEFAULT_BASE = "https://api.ecohash.com/v1"
TTS_VOICE = {"kokoro-82m": "af_heart", "qwen3-tts": "Cherry"}
DEFAULT_VOICE = "af_heart"
TTS_TEXTS = [
    "Hello, how can I help you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Please confirm your appointment for next Tuesday at three in the afternoon.",
    "Our return policy allows refunds within thirty days of purchase, no questions asked.",
    "I'm sorry, but that item is currently out of stock. Would you like me to notify you when it's back?",
    "To reset your password, go to the settings page, click security, and follow the on-screen instructions.",
    "Thank you for calling. Your estimated wait time is approximately four minutes.",
    "The weather tomorrow will be partly cloudy with a high of seventy-two degrees.",
]


def headers(key):
    return {"Authorization": f"Bearer {key}"} if key else {}


def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def to_wav(arr, sr):
    import numpy as np
    pcm = (np.clip(arr, -1, 1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(sr)); w.writeframes(pcm)
    return buf.getvalue()


def wav_duration(b):
    try:
        import soundfile as sf
        arr, sr = sf.read(io.BytesIO(b))
        return len(arr) / float(sr)
    except Exception:
        try:
            with wave.open(io.BytesIO(b)) as w:
                return w.getnframes() / float(w.getframerate())
        except Exception:
            return None


def discover(base, key, verify):
    root = base.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    r = requests.get(root + "/platform/models", headers=headers(key), timeout=30, verify=verify)
    r.raise_for_status()
    raw = r.json()
    models = raw if isinstance(raw, list) else raw.get("data", raw)
    stt, tts = [], []
    for m in models:
        if m.get("status", "active") != "active":
            continue
        cat, mid = m.get("category"), m.get("model_id")
        if cat == "speech_stt":
            stt.append(mid)
        elif cat == "speech_tts":
            tts.append(mid)
    return stt, tts


def load_clips(args):
    # datasets>=4 needs torchcodec to auto-decode audio; avoid that by reading
    # the raw bytes (decode=False) and decoding with soundfile ourselves.
    from datasets import load_dataset, Audio
    import soundfile as sf
    ds = load_dataset(args.dataset, args.config, split=args.split)
    ds = ds.cast_column("audio", Audio(decode=False))
    clips = []
    for row in ds:
        a = row["audio"]
        ref = row.get("text") or row.get("sentence") or row.get("transcription") or ""
        if not ref:
            continue
        b = a.get("bytes")
        if b is None and a.get("path"):
            b = open(a["path"], "rb").read()
        if not b:
            continue
        arr, sr = sf.read(io.BytesIO(b))
        if getattr(arr, "ndim", 1) > 1:
            arr = arr[:, 0]
        clips.append((arr, sr, ref))
        if len(clips) >= args.stt_n:
            break
    return clips


def stt_one(base, key, model, wav_bytes, verify):
    url = base.rstrip("/") + "/audio/transcriptions"
    t0 = time.perf_counter()
    r = requests.post(url, data={"model": model, "response_format": "json"},
                      files={"file": ("clip.wav", wav_bytes, "audio/wav")},
                      headers=headers(key), timeout=120, verify=verify)
    lat = time.perf_counter() - t0
    if r.status_code != 200:
        return None, lat, f"HTTP {r.status_code}"
    try:
        return r.json().get("text", ""), lat, None
    except Exception:
        return r.text, lat, None


def tts_one(base, key, model, text, voice, fmt, verify):
    url = base.rstrip("/") + "/audio/speech"
    t0 = time.perf_counter()
    r = requests.post(url, json={"model": model, "input": text, "voice": voice, "response_format": fmt},
                      headers=headers(key), timeout=120, stream=True, verify=verify)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    first = None; data = b""
    for ch in r.iter_content(4096):
        if ch:
            if first is None:
                first = time.perf_counter() - t0
            data += ch
    total = time.perf_counter() - t0
    dur = wav_duration(data) if fmt == "wav" else None
    return {"ttfa": first, "total": total, "dur": dur, "bytes": len(data)}, None


def run_stt(base, key, models, clips, args, verify):
    rows = []
    for model in models:
        print(f"\n[STT] {model} — warmup {args.warmup}, 测 {len(clips)} 条", flush=True)
        for i in range(args.warmup):
            if clips:
                stt_one(base, key, model, to_wav(*clips[0][:2]), verify)
        refs, hyps, rtfx, errs = [], [], [], []
        for k, (arr, sr, ref) in enumerate(clips, 1):
            wav_b = to_wav(arr, sr); audio_dur = len(arr) / float(sr)
            hyp, lat, err = stt_one(base, key, model, wav_b, verify)
            if err:
                errs.append(err); status = "ERR " + err
            else:
                refs.append(norm(ref)); hyps.append(norm(hyp)); rtfx.append(audio_dur / lat)
                status = "rtfx=%.1f" % (audio_dur / lat)
            print(f"  {k}/{len(clips)} {status}        ", end="\r", flush=True)
            time.sleep(args.sleep)
        wer = None
        if refs:
            import jiwer
            wer = jiwer.wer(refs, hyps) * 100
        rows.append({"model": model, "ok": len(refs), "n": len(clips),
                     "wer": wer, "rtfx_med": statistics.median(rtfx) if rtfx else None,
                     "rtfx_min": min(rtfx) if rtfx else None, "rtfx_max": max(rtfx) if rtfx else None,
                     "err": errs[0] if errs else "", "nerr": len(errs)})
        wer_s = f"{wer:.2f}%" if wer is not None else "n/a"
        rtfx_s = f"{statistics.median(rtfx):.1f}" if rtfx else "n/a"
        print(f"\n  -> WER={wer_s}  RTFx(中位)={rtfx_s}  错误 {len(errs)}/{len(clips)}", flush=True)
    return rows


def run_tts(base, key, models, args, verify):
    rows = []
    texts = TTS_TEXTS[:args.tts_n]
    for model in models:
        voice = TTS_VOICE.get(model, DEFAULT_VOICE)
        print(f"\n[TTS] {model} (voice={voice}) — warmup {args.warmup}, 测 {len(texts)} 句", flush=True)
        for i in range(args.warmup):
            tts_one(base, key, model, texts[0], voice, args.tts_format, verify)
        ttfa, total, rtf, errs = [], [], [], []
        for k, text in enumerate(texts, 1):
            res, err = tts_one(base, key, model, text, voice, args.tts_format, verify)
            if err:
                errs.append(err); status = "ERR " + err
            else:
                if res["ttfa"] is not None:
                    ttfa.append(res["ttfa"] * 1000)
                total.append(res["total"] * 1000)
                if res["dur"]:
                    rtf.append(res["total"] / res["dur"])
                status = "ttfa=%.0fms" % (res["ttfa"] * 1000) if res["ttfa"] is not None else "ok"
            print(f"  {k}/{len(texts)} {status}        ", end="\r", flush=True)
            time.sleep(args.sleep)
        streaming = "否(疑非流式)"
        if ttfa and total and statistics.median(ttfa) < 0.85 * statistics.median(total):
            streaming = "是"
        rows.append({"model": model, "ok": len(total), "n": len(texts),
                     "ttfa_med": statistics.median(ttfa) if ttfa else None,
                     "total_med": statistics.median(total) if total else None,
                     "rtf_med": statistics.median(rtf) if rtf else None,
                     "streaming": streaming, "err": errs[0] if errs else "", "nerr": len(errs)})
        ttfa_s = f"{statistics.median(ttfa):.0f}ms" if ttfa else "n/a"
        rtf_s = f"{statistics.median(rtf):.3f}" if rtf else "n/a"
        print(f"\n  -> TTFA(中位)={ttfa_s}  RTF(中位)={rtf_s}  流式={streaming}  错误 {len(errs)}/{len(texts)}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description="Measure WER / RTFx / TTFA for EcoHash speech models")
    p.add_argument("--api-key", default=os.environ.get("ECOHASH_API_KEY", ""))
    p.add_argument("--base-url", default=DEFAULT_BASE)
    p.add_argument("--models", default="", help="逗号分隔覆盖；默认自动发现平台 active 语音模型")
    p.add_argument("--stt-n", type=int, default=30, help="STT 测试条数")
    p.add_argument("--tts-n", type=int, default=8, help="TTS 测试句数（最多 8）")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--sleep", type=float, default=0.3, help="请求间隔秒，避让 per-key 限流")
    p.add_argument("--tts-format", default="wav", choices=["wav", "mp3"], help="wav 才能算 RTF")
    p.add_argument("--dataset", default="hf-internal-testing/librispeech_asr_dummy")
    p.add_argument("--config", default="clean")
    p.add_argument("--split", default="validation")
    p.add_argument("--verify-ssl", action="store_true")
    p.add_argument("--csv", default="", help="可选：把结果写到这个 CSV")
    args = p.parse_args()
    verify = args.verify_ssl
    if not verify:
        try:
            requests.packages.urllib3.disable_warnings()
        except Exception:
            pass

    if not args.api_key:
        print("缺 API key：--api-key 或 export ECOHASH_API_KEY=eco_xxx", file=sys.stderr); return 1

    if args.models:
        want = [m.strip() for m in args.models.split(",") if m.strip()]
        all_stt, all_tts = discover(args.base_url, args.api_key, verify)
        stt = [m for m in want if m in all_stt]
        tts = [m for m in want if m in all_tts]
        unknown = [m for m in want if m not in all_stt + all_tts]
        if unknown:
            print(f"注意：这些不是平台上的语音模型，跳过：{unknown}", file=sys.stderr)
    else:
        stt, tts = discover(args.base_url, args.api_key, verify)

    print(f"目标：{args.base_url}")
    print(f"STT 模型：{stt or '(无)'}")
    print(f"TTS 模型：{tts or '(无)'}")

    clips = []
    if stt:
        print(f"加载语音数据集 {args.dataset} [{args.config}/{args.split}] …", flush=True)
        try:
            clips = load_clips(args)
            print(f"  取到 {len(clips)} 条带标注语音")
        except Exception as ex:
            print(f"  数据集加载失败：{ex}\n  装一下：pip install --user datasets soundfile numpy", file=sys.stderr)
            stt = []

    stt_rows = run_stt(args.base_url, args.api_key, stt, clips, args, verify) if stt and clips else []
    tts_rows = run_tts(args.base_url, args.api_key, tts, args, verify) if tts else []

    print("\n" + "=" * 78)
    print("STT 结果（WER 越低越好；RTFx 越高越快，是 API 端到端值含网络）")
    print(f"{'模型':24} {'OK/N':>7} {'WER%':>7} {'RTFx中位':>9} {'RTFx范围':>15} 错误")
    for r in stt_rows:
        rng = f"{r['rtfx_min']:.1f}-{r['rtfx_max']:.1f}" if r["rtfx_min"] else "-"
        wer_s = f"{r['wer']:.2f}" if r["wer"] is not None else "n/a"
        rtfx_s = f"{r['rtfx_med']:.1f}" if r["rtfx_med"] else "n/a"
        print(f"{r['model']:24} {r['ok']}/{r['n']:<5} {wer_s:>7} {rtfx_s:>9} {rng:>15} {r['nerr']} {r['err']}")
    print("\nTTS 结果（TTFA 越低越好；RTF<1 才能流式实时）")
    print(f"{'模型':24} {'OK/N':>7} {'TTFA中位ms':>11} {'总耗时中位ms':>13} {'RTF中位':>9} {'流式':>10} 错误")
    for r in tts_rows:
        ttfa_s = f"{r['ttfa_med']:.0f}" if r["ttfa_med"] else "n/a"
        total_s = f"{r['total_med']:.0f}" if r["total_med"] else "n/a"
        rtf_s = f"{r['rtf_med']:.3f}" if r["rtf_med"] else "n/a"
        print(f"{r['model']:24} {r['ok']}/{r['n']:<5} {ttfa_s:>11} {total_s:>13} {rtf_s:>9} {r['streaming']:>10} {r['nerr']} {r['err']}")
    print("=" * 78)

    if args.csv:
        import csv as csvmod
        with open(args.csv, "w", newline="") as f:
            w = csvmod.writer(f)
            w.writerow(["type", "model", "ok", "n", "wer", "rtfx_med", "ttfa_med_ms", "total_med_ms", "rtf_med", "streaming", "errors"])
            for r in stt_rows:
                w.writerow(["stt", r["model"], r["ok"], r["n"], r["wer"], r["rtfx_med"], "", "", "", "", r["nerr"]])
            for r in tts_rows:
                w.writerow(["tts", r["model"], r["ok"], r["n"], "", "", r["ttfa_med"], r["total_med"], r["rtf_med"], r["streaming"], r["nerr"]])
        print(f"已写入 {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

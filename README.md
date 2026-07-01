<p align="center">
  <a href="https://ecohash.com"><img src="assets/ecohash-logo.png" alt="EcoHash" width="280"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT">
  <a href="https://docs.ecohash.com"><img src="https://img.shields.io/badge/documentation-6D28D9" alt="Documentation"></a>
  <a href="https://x.com/ecohashdev"><img src="https://img.shields.io/badge/X-@ecohashdev-000000?logo=x&logoColor=white" alt="X"></a>
  <a href="https://huggingface.co/ecohash-ai"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ecohash--ai-FFD21E" alt="Hugging Face"></a>
</p>

# EcoHash Benchmarks

Open, reproducible performance benchmarks for the open models served on EcoHash — an OpenAI-compatible inference API.

<p align="center">
  <img src="assets/stt-wer-vs-rtfx.png" alt="Speech-to-text: WER vs RTFx" width="49%">
  <img src="assets/tts-ttfa.png" alt="Text-to-speech: time to first audio" width="49%">
</p>

## Methodology

The same open model gives different numbers depending on **who serves it and how it's measured**. Each table below lists open models with a **Source** column; **EcoHash rows (in bold) are our own measurements**, as one provider among others.

- **Open ASR Leaderboard (A100)** — WER (English 8-dataset average) and RTFx from the [HF Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard): all models on a single A100-80GB, batch 64. That RTFx is A100 throughput, not our hardware.
- **EcoHash (end-to-end)** — measured client-side through our API (`https://api.ecohash.com/v1`) on **NVIDIA RTX PRO 6000**, so it includes network + queueing. STT WER is on LibriSpeech (clean, English, simple normalization). Measured 2026-06-23.

> Numbers from different sources are **not apples-to-apples** (A100 batch throughput vs single-stream end-to-end; different WER sets) — the Source column says how each row was produced. Prices are per-provider and indicative. A pure-GPU EcoHash rerun with more models is planned.

## Speech-to-text

**Bold = measured by EcoHash.**

| Model | Source | WER % ↓ | RTFx ↑ | Price |
|---|---|---|---|---|
| **whisper-large-v3** | **EcoHash (end-to-end)** | **7.26** | **3.0** | **$0.006/min** |
| **qwen3-asr-1-7b** | **EcoHash (end-to-end)** | **4.36** | **20.0** | **input $0.05/1M tok** |
| **fun-asr-nano** | **EcoHash (end-to-end)** | **4.00** | **12.3** | **input $0.05/1M tok** |
| CohereLabs/cohere-transcribe-03-2026 | Leaderboard (A100) | 5.42 | 525 | — |
| ibm-granite/granite-4.0-1b-speech | Leaderboard (A100) | 5.52 | 280 | — |
| nvidia/canary-qwen-2.5b | Leaderboard (A100) | 5.63 | 418 | $0.00074/min (Replicate) |
| ibm-granite/granite-speech-3.3-8b | Leaderboard (A100) | 5.74 | 145 | — |
| Qwen/Qwen3-ASR-1.7B | Leaderboard (A100) | 5.76 | 148 | — |
| ibm-granite/granite-speech-3.3-2b | Leaderboard (A100) | 6.00 | 271 | — |
| microsoft/Phi-4-multimodal-instruct | Leaderboard (A100) | 6.02 | 151 | — |
| nvidia/parakeet-tdt-0.6b-v2 | Leaderboard (A100) | 6.05 | 3386 | — |
| nvidia/parakeet-tdt-0.6b-v3 | Leaderboard (A100) | 6.32 | 3333 | $0.0015/min (Together) |
| nvidia/canary-1b-flash | Leaderboard (A100) | 6.35 | 1046 | — |
| kyutai/stt-2.6b-en | Leaderboard (A100) | 6.40 | 88 | — |
| Qwen/Qwen3-ASR-0.6B | Leaderboard (A100) | 6.42 | 166 | — |
| mistralai/Voxtral-Small-24B-2507 | Leaderboard (A100) | 6.62 | 54 | $0.004/min (Mistral) |
| nvidia/parakeet-tdt-1.1b | Leaderboard (A100) | 7.02 | 2391 | — |
| mistralai/Voxtral-Mini-3B-2507 | Leaderboard (A100) | 7.05 | 110 | $0.001/min (DeepInfra) |
| nvidia/parakeet-ctc-1.1b | Leaderboard (A100) | 7.40 | 2729 | — |
| openai/whisper-large-v3 | Leaderboard (A100) | 7.44 | 146 | from $0.00045/min |
| nvidia/parakeet-tdt_ctc-110m | Leaderboard (A100) | 7.49 | 5345 | — |
| distil-whisper/distil-large-v3 | Leaderboard (A100) | 7.52 | 214 | — |
| openai/whisper-large-v3-turbo | Leaderboard (A100) | 7.83 | 200 | $0.003/min (OpenAI 4o-mini) |
| usefulsensors/moonshine-streaming-small | Leaderboard (A100) | 7.84 | 566 | — |
| usefulsensors/moonshine-base | Leaderboard (A100) | 9.99 | 566 | — |

> **WER is not comparable across sources.** EcoHash rows are measured on LibriSpeech test-clean (the easiest English set); leaderboard rows are the 8-dataset average, which includes much harder audio. The same model scores a lower WER on clean-only audio — so EcoHash's `whisper-large-v3` at 7.26 is **not** better than the leaderboard's 7.44 for the same model; it is the easier benchmark. Compare WER only within the same Source column, and read RTFx the same way (A100 batch-64 vs single-stream end-to-end).

Full data (params, notes): [speech/stt.csv](speech/stt.csv).

## Text-to-speech

**Bold = measured by EcoHash.** There is no unified TTS speed leaderboard, so open-model TTFA is measured where available, otherwise vendor-claimed.

| Model | Source | TTFA | Streaming | Price |
|---|---|---|---|---|
| **kokoro-82m** | **EcoHash (end-to-end)** | **325 ms** | **no** | **$0.50/1M** |
| **qwen3-tts** | **EcoHash (end-to-end)** | **3538 ms** | **no** | **$10/1M** |
| Kokoro-82M | Open model (hexgrad) | ~325 ms (measured) | no | $0.62/1M (DeepInfra) |
| Chatterbox-0.5B | Open model (Resemble) | ~472 ms (measured) | yes | — |
| MeloTTS | Open model (MyShell) | non-streaming | no | — |
| Orpheus-3B | Open model (Canopy) | ~200 ms (claimed) | yes | $15/1M (Together) |
| CosyVoice2-0.5B | Open model (Alibaba) | ~150 ms (claimed) | yes | $7.15/1M (SiliconFlow) |
| Qwen3-TTS | Open model (Alibaba) | 97 ms (claimed) | yes | ~CNY 80/1M (DashScope) |
| Parler-TTS-large | Open model (Hugging Face) | <500 ms (claimed) | yes | — |

Full data (params, notes): [speech/tts.csv](speech/tts.csv).

## Reproduce

Our EcoHash (end-to-end) numbers:

```bash
pip install openai jiwer datasets soundfile numpy requests
export ECOHASH_API_KEY=eco_...   # create one at console.ecohash.com
python speech/benchmark.py --stt-n 50 --tts-n 8
```

The charts:

```bash
pip install matplotlib
python speech/plot.py
```

Leaderboard numbers are sourced from the [HF Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

## License

MIT — see [LICENSE](LICENSE). Data may be reused with attribution.

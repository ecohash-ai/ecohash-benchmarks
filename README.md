<p align="center">
  <img src="assets/ecohash-logo.png" alt="EcoHash" width="280">
</p>

<p align="center">
  <a href="https://console.ecohash.com?utm_source=github">Platform</a> •
  <a href="https://docs.ecohash.com">Docs</a> •
  <a href="https://ecohash.com">Website</a>
</p>

# EcoHash Benchmarks

Open, reproducible performance benchmarks for the open models served on EcoHash — an OpenAI-compatible inference API.

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT">
  <img src="https://img.shields.io/badge/open%20speech%20models-29-6D28D9" alt="29 open speech models">
  <img src="https://img.shields.io/badge/measured-2026--06--23-lightgrey" alt="measured 2026-06-23">
</p>

<p align="center">
  <img src="assets/stt-wer-vs-rtfx.png" alt="Speech-to-text: WER vs RTFx" width="49%">
  <img src="assets/tts-ttfa.png" alt="Text-to-speech: time to first audio" width="49%">
</p>

## Methodology

Two kinds of numbers, kept separate on purpose:

**1. Open-model landscape (reference).** WER and RTFx in the STT landscape come from the [HF Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — all models run on a single **A100-80GB, batch 64**, and WER is the English 8-dataset average. That RTFx is an **A100** figure, not RTX PRO 6000 and not our measurement. The TTS landscape lists popular open models by HF downloads; TTFA is **measured where available, otherwise vendor/paper-claimed** (noted per row).

**2. Measured on EcoHash (end-to-end).** The "measured on EcoHash" tables are our own numbers, measured **client-side end-to-end through the API** (`https://api.ecohash.com/v1`) on **NVIDIA RTX PRO 6000** endpoints, so they include network + queueing. STT: WER via [jiwer](https://github.com/jitsi/jiwer) on LibriSpeech (clean, English, simple normalization), RTFx = audio ÷ latency. TTS: TTFA = time to first audio byte, RTF = total time ÷ audio duration. Warmup + median across samples. Measured **2026-06-23** (STT: 50 clips/model; TTS: 8 sentences/model).

> **Caveats.** Our end-to-end RTFx/TTFA are **not comparable** to the A100 leaderboard (short clips make network + overhead dominant, so RTFx reads low); a pure-GPU rerun with more models is planned. WER on LibriSpeech-clean is lower than the leaderboard's 8-dataset average and uses a simple normalizer — for cross-model comparison only.

## Speech-to-text — open-model landscape

Source: HF Open ASR Leaderboard (A100, batch 64; WER = English 8-dataset average). **✅ = served on EcoHash.**

| # | Model | Params (B) | WER % ↓ | RTFx (A100) ↑ | On EcoHash |
|---|---|---|---|---|---|
| 1 | CohereLabs/cohere-transcribe-03-2026 | 2 | 5.42 | 525 | |
| 3 | ibm-granite/granite-4.0-1b-speech | 2 | 5.52 | 280 | |
| 4 | nvidia/canary-qwen-2.5b | 2.5 | 5.63 | 418 | |
| 5 | ibm-granite/granite-speech-3.3-8b | 9 | 5.74 | 145 | |
| 6 | **Qwen/Qwen3-ASR-1.7B** | 1.7 | 5.76 | 148 | ✅ |
| 8 | ibm-granite/granite-speech-3.3-2b | 3 | 6.00 | 271 | |
| 9 | microsoft/Phi-4-multimodal-instruct | 6 | 6.02 | 151 | |
| 10 | nvidia/parakeet-tdt-0.6b-v2 | 0.6 | 6.05 | 3386 | |
| 13 | nvidia/parakeet-tdt-0.6b-v3 | 0.6 | 6.32 | 3333 | |
| 14 | nvidia/canary-1b-flash | 1 | 6.35 | 1046 | |
| 15 | kyutai/stt-2.6b-en | 2.6 | 6.40 | 88 | |
| 17 | Qwen/Qwen3-ASR-0.6B | 0.6 | 6.42 | 166 | |
| 19 | mistralai/Voxtral-Small-24B-2507 | 24 | 6.62 | 54 | |
| 26 | nvidia/parakeet-tdt-1.1b | 1.1 | 7.02 | 2391 | |
| 28 | mistralai/Voxtral-Mini-3B-2507 | 5 | 7.05 | 110 | |
| 35 | nvidia/parakeet-ctc-1.1b | 1.1 | 7.40 | 2729 | |
| 38 | **openai/whisper-large-v3** | 2 | 7.44 | 146 | ✅ |
| 39 | nvidia/parakeet-tdt_ctc-110m | 0.11 | 7.49 | 5345 | |
| 41 | distil-whisper/distil-large-v3 | 0.8 | 7.52 | 214 | |
| 46 | openai/whisper-large-v3-turbo | 0.8 | 7.83 | 200 | |
| 48 | usefulsensors/moonshine-streaming-small | 0.12 | 7.84 | 566 | |
| 66 | usefulsensors/moonshine-base | 0.06 | 9.99 | 566 | |

Full data (prices, notes): [speech/stt.csv](speech/stt.csv).

## Speech-to-text — measured on EcoHash (end-to-end via API)

| Model | WER % (LibriSpeech clean, en) | RTFx (end-to-end) |
|---|---|---|
| `whisper-large-v3` | 7.26 | 3.0 |
| `qwen3-asr-1-7b` | 4.36 | 20.0 |
| `fun-asr-nano` | 4.00 | 12.3 |

Data: [speech/stt-ecohash.csv](speech/stt-ecohash.csv).

## Text-to-speech — open-model landscape

Popular open TTS models by HF downloads. There is no unified TTS speed leaderboard, so TTFA is measured where available, otherwise vendor-claimed. **✅ = served on EcoHash.**

| Model | Provider | Params | TTFA | Streaming | On EcoHash |
|---|---|---|---|---|---|
| **Kokoro-82M** | hexgrad | 82M | ~325 ms (measured) | no | ✅ |
| Chatterbox-0.5B | Resemble | 0.5B | ~472 ms (measured) | yes | |
| MeloTTS | MyShell | ~52M | non-streaming | no | |
| Orpheus-3B | Canopy | 3B | ~200 ms (claimed) | yes | |
| CosyVoice2-0.5B | Alibaba | 0.5B | ~150 ms (claimed) | yes | |
| **Qwen3-TTS** | Alibaba | 1.7B/0.6B | 97 ms (claimed) | yes | ✅ |
| Parler-TTS-large | Hugging Face | 2.2B | <500 ms (claimed) | yes | |

Full data (prices, notes): [speech/tts.csv](speech/tts.csv).

## Text-to-speech — measured on EcoHash (end-to-end via API)

| Model | TTFA (end-to-end) | RTF | Streaming |
|---|---|---|---|
| `kokoro-82m` | 325 ms | 0.077 | no |
| `qwen3-tts` | 3538 ms | 0.593 | no |

Data: [speech/tts-ecohash.csv](speech/tts-ecohash.csv).

## Reproduce

Our "measured on EcoHash" numbers:

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

Landscape numbers are sourced from the [HF Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

## License

MIT — see [LICENSE](LICENSE). Data may be reused with attribution.

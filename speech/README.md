# Speech benchmarks

End-to-end (via API) quality and speed for EcoHash's speech models.

- `stt.csv` — transcription: WER + RTFx (end-to-end)
- `tts.csv` — synthesis: TTFA + RTF (end-to-end)
- `benchmark.py` — the measurement script; reproduce with your own API key

See the [top-level README](../README.md) for methodology, result tables, and caveats.

```bash
pip install openai jiwer datasets soundfile numpy
export ECOHASH_API_KEY=eco_...
python benchmark.py --stt-n 50 --tts-n 8
```

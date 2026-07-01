# Speech benchmarks

- `stt.csv` — transcription: the Open ASR Leaderboard landscape + EcoHash end-to-end rows
- `tts.csv` — synthesis: popular open models + EcoHash end-to-end rows
- `benchmark.py` — measure the EcoHash (end-to-end) numbers with your own API key
- `plot.py` — regenerate the charts from the CSVs

See the [top-level README](../README.md) for the result tables, methodology, and caveats.

```bash
pip install openai jiwer datasets soundfile numpy requests matplotlib
export ECOHASH_API_KEY=eco_...
python benchmark.py --stt-n 50 --tts-n 8   # measure EcoHash end-to-end
python plot.py                             # regenerate charts
```

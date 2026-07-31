# Phase-Based Validation Analysis

- records: `data/processed/structured_11x11_expanded_5000_imitation_records.jsonl`
- split: `validation`
- checkpoint: `models/final/imitation_11x11_best.pt`

| Phase | Examples | Loss | Top-1 accuracy | Top-5 accuracy |
|---|---:|---:|---:|---:|
| opening | 11877 | 1.0944 | 0.7282 | 0.9267 |
| midgame | 6137 | 1.0611 | 0.7116 | 0.9492 |
| endgame | 2978 | 1.5448 | 0.6108 | 0.9073 |

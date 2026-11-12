# Benchmark plan: Qwen3-VL on Action Genome (frame-level scene graphs)

Goal: one fixed protocol so every model is scored on the same frames, same prompt, same metric.
Status: plan only (2026-11-12). No result below this line has been produced under this protocol.

## Test set
- `data/test.jsonl`: 280 frames from **35 videos** (8 frames each), built by `prepare_data.py --split test` (Action Genome `set == "test"`). Training used `--split train` (500 Charades videos), so the splits should be disjoint; confirm by listing the training IDs and intersecting.
- The old README says "25 test videos"; the file has 35. Use 35 / 280 and record the video IDs with every result.
- Not in git (`data/` is ignored; Action Genome and Charades are non-commercial research data). Copy `data/test.jsonl`, `data/img/` and `vocab.json` by hand, or regenerate from the annotation pickle + Charades videos (README steps 1-4).
- `eval_gguf.py` is currently untracked: copy it as well.

## Models
Ours (Hugging Face, account Ram20307): `qwen3vl-2b-ag-lora`, `qwen3vl-2b-ag-merged` (bf16), `qwen3vl-2b-ag-GGUF` (Q4_K_M + f16 mmproj, run with `-c 4096`). `ag-hico` variants optional; `hico`-only scores 0 on this set (does not know the AG label format), skip.
Base (`Qwen/`): Qwen3-VL-2B-Instruct, 4B, 8B, and 30B-A3B or 32B if VRAM allows (teacher candidates).

## Runs
| # | Model | Scorer | Why |
|---|---|---|---|
| 1 | ag fine-tune, bf16 merged | `eval_ag.py` | Settles quantisation cost (Q4 is .30/.52 on these frames; bf16 .31/.45 was on different frames) |
| 2 | ag fine-tune, Q4_K_M | `eval_gguf.py --tag ag_q4` | Like-for-like with run 1 |
| 3 | base 2B / 4B / 8B, zero-shot | `eval_gguf.py --vocab vocab.json` | Fair baseline; without the label list they score 0 (format, not vision). Earlier 4B zero-shot was unusable even with the vocabulary: report that. |
| 4 | base 30B-A3B / 32B, zero-shot with `--vocab` | same | Teacher gap: does the bigger model beat our 2B fine-tune zero-shot? |
| 5 | optional: 30B-A3B, few-shot prompt | same | How much of the gap is prompting |

## Rules
- Same 280 frames, same prompt, temperature 0, nothing else on the GPU, same context length per backend.
- Report per run: triple recall / precision / F1, predicate-only recall / precision, JSON-valid rate, seconds per frame, peak VRAM.
- Report two triple metrics: (a) all triples (current `eval_gguf.py`), (b) **hard** = excluding `not_looking_at` and `not_contacting`, which is the metric the Sanjaya harness uses (held-out recall .17 / precision .44) and is not inflated by easy negatives.
- Temperature 0 repeats are near-identical; do not report spread from repeats. Say the variance is in the data (small test set).
- Report failures: zero-recall predicates (`unsure`, etc.), invalid JSON cases.
- Do not claim SOTA, do not compare to systems we did not run, and call results preliminary (35 videos).

## Known numbers so far (not under this protocol, do not quote as a benchmark)
ag fine-tune Q4_K_M, 280 frames: R .30 / P .52, predicate-only .62 / .83, 1.6 s/frame on an RTX 2070. ag-hico Q4_K_M: R .28 / P .52, 0.61 s/frame.

## Outputs
`results_<tag>.json` per run plus one table in this file. Record: model, quant, backend, GPU, driver, prompt hash, video IDs.

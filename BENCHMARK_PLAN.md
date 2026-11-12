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

## Datasets for the test (all in scope; licence = non-commercial research unless stated)
| Dataset | Used for | Size / status | Gold |
|---|---|---|---|
| **Action Genome** (Ji et al., CVPR 2020) | Main frame-level test: `data/test.jsonl` | 280 frames, 35 test-split videos, 8 frames each. Training used its `train` split (500 videos, ~3.9k frames). | Human annotation (attention / spatial / contact) |
| **Charades** (Allen AI) | Video source for Action Genome frames and for the Sanjaya demo drops | 480p videos, only the selected ids | n/a (videos only) |
| **STAR** (Wu et al., MIT-IBM) | Sanjaya hand-annotated clips FL6DF, R9382, 3T785 (Charades videos) | 3 clips, temporal graphs | Hand-made, written from frames (3T785 sampled every 2.5 s) |
| **Action Genome demo clips** FVINY, 9UU4W, 2KGV3, 1E1EN, QIT2W | Temporal triple recall/precision in the Sanjaya harness (`ag_bench.py`) | 5 clips; confirm none are in the training ids | Human annotation |
| **Own hand-made clips** test17 (handshake), test20 (4-person meeting) | Temporal scene graph vs hand-made graph (`harness.py --eval`) | 2 clips, source not recorded (stock footage, to be confirmed) | Hand-made |
| **Unseen live-drop clips** 8D4ZT, WT46G, T3R3K, GKH0F, TK76G, 1ZBUS, JOLLV | End-to-end live pipeline (time, valid output) | 7 clips, 12 s each, not gold-scored | none |
| **HICO / HICO-DET** | Only the `hico` fine-tune variants | Numbers come from the Hugging Face cards (bf16). Test set not downloaded here, not run. | HICO annotation |
| **CUHK Avenue** | Planned: anomaly-event evaluation of the Sanjaya harness | Listed as downloaded in the Sanjaya pitch note; not part of this frame-level benchmark | Anomaly labels |

Report each dataset separately; never pool hand-made and human-annotated scores.

## Teacher test (2026-11-12, single run, preliminary)
Gemini as a teacher on test20 (hand-made gold, harness `score` / `score2`, predicates restricted to the harness vocabulary): `gemini-3.1-flash-lite` recall/precision 0.79 / 1.0 (IoU), coverage 0.79 / 0.80, 3.1 s for the 15.9 s clip; `gemini-3.5-flash` 0.50 / 1.0, coverage 0.50 / 0.71, 9.9 s. One clip, one run, gold is small. Not comparable to the frame-level AG numbers above. Output JSON: `sanjaya/eval/gemini/`.

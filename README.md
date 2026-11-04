# ag-finetune: LoRA Qwen3-VL-2B on Action Genome (16 GB GPU)
Transfer from the demo laptop: `object_bbox_and_relationship.pkl` (~292 MB) and `videos/` (Charades 480p, only the ids you picked). Nothing else.
1. `python prepare_data.py --ann object_bbox_and_relationship.pkl --list 500 > train_ids.txt`
2. `python ag_fetch.py $(cat train_ids.txt)`  (range-fetches just those videos from the Charades zip into ag_data/videos; then `mv ag_data/videos videos`)
3. `python prepare_data.py --ann object_bbox_and_relationship.pkl --videos videos --split train --out data/train.jsonl`
4. same with `--split test --out data/test.jsonl` (use the 25 test videos from the demo laptop)
5. Baseline: `python eval_ag.py --data data/test.jsonl`
6. `python train.py --data data/train.jsonl` then `python eval_ag.py --data data/test.jsonl --adapter out/qwen3vl2b-ag`
Compare against the harness numbers (held-out recall .17 / precision .44 triple-level; looking_at 0, carrying 0).
Not tested end to end on the 16 GB machine: check VRAM on a 50-sample `--max-samples 50` run first, raise `--bs` if there is headroom.

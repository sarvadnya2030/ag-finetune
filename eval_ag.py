"""Score a model on held-out AG test frames: per-predicate recall/precision (contact, attention) + JSON-valid rate.
  python eval_ag.py --data data/test.jsonl [--adapter out/qwen3vl2b-ag]      # omit --adapter for the base-model baseline"""
import argparse, collections, json, re, torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--adapter", default=""); ap.add_argument("--limit", type=int, default=300); a = ap.parse_args()
proc = AutoProcessor.from_pretrained(a.model); model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto")
if a.adapter:
    from peft import PeftModel; model = PeftModel.from_pretrained(model, a.adapter)
model.eval(); tp, fp, fn, ok = collections.Counter(), collections.Counter(), collections.Counter(), 0
rows = [json.loads(l) for l in open(a.data)][:a.limit]
def triples(js): return {(o["object"], f"{k}:{p}") for o in js for k in ("attention", "contact") for p in o.get(k, [])}
for r in rows:
    m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["prompt"]}]}]
    x = proc(text=[proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)], images=[Image.open(r["image"]).convert("RGB")], return_tensors="pt").to(model.device)
    with torch.no_grad(): y = model.generate(**x, max_new_tokens=400, do_sample=False)
    txt = proc.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    try: pred = triples(json.loads(re.search(r"\[.*\]", txt, re.S).group(0))); ok += 1
    except Exception: pred = set()
    gold = triples(json.loads(r["answer"]))
    for t in pred & gold: tp[t[1]] += 1
    for t in pred - gold: fp[t[1]] += 1
    for t in gold - pred: fn[t[1]] += 1
print(f"json valid {ok}/{len(rows)}"); T, P, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
print(f"ALL  R {T/max(T+N,1):.2f}  P {T/max(T+P,1):.2f}")
for p in sorted(set(tp) | set(fn), key=lambda p: -(tp[p] + fn[p]))[:15]: print(f"{p:28s} R {tp[p]/max(tp[p]+fn[p],1):.2f}  P {tp[p]/max(tp[p]+fp[p],1):.2f}  n={tp[p]+fn[p]}")

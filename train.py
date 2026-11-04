"""LoRA fine-tune Qwen3-VL-2B on the JSONL from prepare_data.py (16 GB GPU: bf16, no quantisation).
  python train.py --data data/train.jsonl --out out/qwen3vl2b-ag [--model Qwen/Qwen3-VL-2B-Instruct --epochs 2]"""
import argparse, json, torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="out/qwen3vl2b-ag")
ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct"); ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--bs", type=int, default=2); ap.add_argument("--accum", type=int, default=8); ap.add_argument("--max-samples", type=int, default=0); a = ap.parse_args()
proc = AutoProcessor.from_pretrained(a.model); proc.tokenizer.padding_side = "right"
model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto")
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)", task_type="CAUSAL_LM"))
rows = [json.loads(l) for l in open(a.data)][:a.max_samples or None]
def msgs(r, ans=True):
    m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["prompt"]}]}]
    return m + ([{"role": "assistant", "content": [{"type": "text", "text": r["answer"]}]}] if ans else [])
def collate(batch):
    full = [proc.apply_chat_template(msgs(r), tokenize=False) for r in batch]; imgs = [Image.open(r["image"]).convert("RGB") for r in batch]
    enc = proc(text=full, images=imgs, return_tensors="pt", padding=True)
    lab = enc["input_ids"].clone(); lab[enc["attention_mask"] == 0] = -100
    for i, r in enumerate(batch):  # loss on the answer only: mask the prompt prefix
        n = proc(text=[proc.apply_chat_template(msgs(r, False), tokenize=False, add_generation_prompt=True)], images=[imgs[i]], return_tensors="pt")["input_ids"].shape[1]
        lab[i, :n] = -100
    enc["labels"] = lab; return enc
Trainer(model=model, data_collator=collate, train_dataset=rows, args=TrainingArguments(
    output_dir=a.out, per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum, num_train_epochs=a.epochs, learning_rate=a.lr,
    lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True, logging_steps=10, save_strategy="epoch", save_total_limit=2, remove_unused_columns=False, report_to="none")).train()
model.save_pretrained(a.out); proc.save_pretrained(a.out); print("saved", a.out)

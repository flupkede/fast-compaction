# QLoRA fine-tune of a small student judge from fine-tune/train.jsonl.
# Tuned for an 8 GB GPU (RTX 3060 Ti): 1.7B at seq 8192 fits; 4B needs
# --seq-len 6144. Unsloth currently wants Python 3.11/3.12 in a venv.
#
#   pip install unsloth
#   python train-student.py --data fine-tune/train.jsonl --out student-out
#
# Output: LoRA adapters + merged GGUF (q8_0), then on the ollama box:
#   ollama create student -f Modelfile  (paths printed at the end)

import argparse
import json
import os

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

DEFAULT_MODEL = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"


def load_examples(path: str) -> list[list[dict]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line)["messages"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="unsloth 4-bit base (Qwen3-1.7B default, Qwen3-4B for --seq-len 6144)")
    ap.add_argument("--data", default="fine-tune/train.jsonl")
    ap.add_argument("--val", default="fine-tune/val.jsonl")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="student-out")
    ap.add_argument("--gguf", default="q8_0", help="GGUF quant, empty string to skip")
    args = ap.parse_args()

    rows = load_examples(args.data)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.seq_len, load_in_4bit=True)
    tpl = tokenizer.apply_chat_template

    def fmt(r):
        return {"text": tpl(r["messages"], tokenize=False)}

    ds = Dataset.from_list([{"messages": r} for r in rows]).map(fmt)
    if args.val and os.path.exists(args.val):
        val = Dataset.from_list([{"messages": r} for r in load_examples(args.val)]).map(fmt)
    else:
        val = None

    # skip-long must run against the templated length, not the raw text
    def fit(ex):
        return len(tokenizer(ex["text"])["input_ids"]) <= args.seq_len

    before = len(ds)
    ds = ds.filter(fit)
    skipped = before - len(ds)
    if val is not None:
        before_v = len(val)
        val = val.filter(fit)
        print(f"skipped over-length examples: train {skipped}, val {before_v - len(val)}")

    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)

    trainer = SFTTrainer(
        model=model, train_dataset=ds, eval_dataset=val,
        processing_class=tokenizer,
        args=SFTConfig(
            per_device_train_batch_size=1, gradient_accumulation_steps=16,
            num_train_epochs=args.epochs, learning_rate=args.lr,
            lr_scheduler_type="cosine", warmup_steps=30, logging_steps=10,
            eval_strategy="steps" if val else "no", eval_steps=100 if val else None,
            save_strategy="no", bf16=True, seed=3407,
            optim="paged_adamw_8bit", output_dir=args.out, report_to="none",
            max_length=args.seq_len, dataset_text_field="text"))
    trainer = train_on_responses_only(
        trainer, instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n")
    trainer.train()
    if val:
        trainer.evaluate()

    model.save_pretrained(os.path.join(args.out, "adapter"))
    tokenizer.save_pretrained(os.path.join(args.out, "adapter"))
    if args.gguf:
        try:
            model.save_pretrained_gguf(args.out, tokenizer, quantization_method=args.gguf)
            gguf = os.path.join(args.out, f"{args.model.split('/')[-1]}.{args.gguf}.gguf")
            print("next: ollama create student -f Modelfile with")
            print(f"  FROM {gguf}\n  PARAMETER num_ctx 16384")
        except Exception as e:  # Windows llama.cpp builds are flaky; convert manually
            print(f"GGUF export failed ({e}); convert the merged model with "
                  "llama.cpp convert_hf_to_gguf.py instead")


if __name__ == "__main__":
    main()

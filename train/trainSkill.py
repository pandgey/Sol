import torch, yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

tok = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
tok.pad_token = tok.eos_token
tok.padding_side = "right"

def format_chat(ex):
    text = tok.apply_chat_template(ex["messages"], tokenize=False)
    return {"text": text}

ds = load_dataset("json", data_files=cfg["dataset_path"], split="train")
ds = ds.map(format_chat, remove_columns=["messages"])

model = AutoModelForCausalLM.from_pretrained(
    cfg["base_model"],
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)

peft_cfg = LoraConfig(
    r=cfg["lora_r"],
    lora_alpha=cfg["lora_alpha"],
    lora_dropout=cfg["lora_dropout"],
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

train_cfg = SFTConfig(
    output_dir=cfg["output_dir"],
    num_train_epochs=cfg["num_train_epochs"],
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
    learning_rate=cfg["learning_rate"],
    lr_scheduler_type=cfg["lr_scheduler_type"],
    warmup_ratio=cfg["warmup_ratio"],
    max_seq_length=cfg["max_seq_length"],
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    logging_steps=25,
    save_steps=1000,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    peft_config=peft_cfg,
    train_dataset=ds,
    dataset_text_field="text",
    args=train_cfg,
)
trainer.train()
trainer.model.save_pretrained(cfg["output_dir"])
tok.save_pretrained(cfg["output_dir"])
import os, torch, yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training

# ──────────────────────────────────────────────
# Load configuration
# ──────────────────────────────────────────────
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

base_model = cfg["base_model"]
dataset_path = cfg["dataset_path"]
output_dir = cfg["output_dir"]

# ──────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tok.pad_token = tok.eos_token
tok.padding_side = "right"

def format_chat(examples):
    """Manually format chat messages into a conversation-style prompt."""
    text = ""
    for msg in examples["messages"]:
        role = msg["role"].capitalize()
        if role == "System":
            text += f"<|system|>: {msg['content'].strip()}\n"
        elif role == "User":
            text += f"<|user|>: {msg['content'].strip()}\n"
        elif role == "Assistant":
            text += f"<|assistant|>: {msg['content'].strip()}\n"
    text += "<|assistant|>: "  # generation prompt
    return {"text": text}

print("Loading dataset:", dataset_path)
ds = load_dataset("json", data_files=dataset_path, split="train")
ds = ds.map(format_chat, remove_columns=["messages"])

# ──────────────────────────────────────────────
# Base model
# ──────────────────────────────────────────────
print("Loading base model:", base_model)
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Load model with proper device mapping
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    dtype=dtype,
    device_map="auto",
    low_cpu_mem_usage=True,
    use_cache=False,  # Disable KV cache to save memory
    torch_dtype=dtype,
    offload_folder="offload",  # Use disk offloading if needed
)

# Prepare model for training with LoRA
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

# ──────────────────────────────────────────────
# LoRA configuration
# ──────────────────────────────────────────────
peft_cfg = LoraConfig(
    r=cfg.get("lora_r", 16),
    lora_alpha=cfg.get("lora_alpha", 32),
    lora_dropout=cfg.get("lora_dropout", 0.05),
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

# ──────────────────────────────────────────────
# Training configuration
# ──────────────────────────────────────────────
train_cfg = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=int(cfg.get("num_train_epochs", 2)),
    per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
    gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 16)),
    gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
    learning_rate=float(cfg.get("learning_rate", 2e-4)),
    lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
    warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    logging_steps=25,
    save_strategy="epoch",
    report_to="none",
    optim="paged_adamw_8bit",  # Use 8-bit optimizer to save memory
    max_grad_norm=0.3,
)

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
print("Starting fine-tuning...")

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    peft_config=peft_cfg,
    args=train_cfg,
    formatting_func=lambda x: x["text"],
)

trainer.train()
trainer.model.save_pretrained(output_dir)
tok.save_pretrained(output_dir)
print(f"Training complete! Adapter saved to {output_dir}")
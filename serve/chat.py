from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", type=str, default="../skills/behavior.json")
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.adapter)
model = AutoModelForCausalLM.from_pretrained(
    args.adapter,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

while True:
    user = input("🧑 You: ")
    if user.strip().lower() in {"exit","quit"}:
        break
    msgs = [
        {"role":"system","content":"You are Sol, a friendly, intelligent anime-style assistant."},
        {"role":"user","content":user}
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    model.generate(**ids, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True, streamer=streamer)
    print()
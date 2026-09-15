import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = r".\models\Qwen3-8B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
start_load = time.perf_counter()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

load_time = time.perf_counter() - start_load

print(f"\n✅ Model loaded in {load_time:.2f} seconds")
print("Device map:")
print(model.hf_device_map)

if torch.cuda.is_available():
    print(
        f"\nGPU memory allocated: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

prompt = "Explain what LLM quantization is in simple terms."

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

# Send inputs to the device where the model's input embeddings live
input_device = model.get_input_embeddings().weight.device
inputs = {k: v.to(input_device) for k, v in inputs.items()}

print("\nGenerating...")
start = time.perf_counter()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )

total_time = time.perf_counter() - start

input_tokens = inputs["input_ids"].shape[-1]
output_tokens = outputs.shape[-1] - input_tokens

print("\n===== BASELINE =====")
print("Input tokens :", input_tokens)
print("Output tokens:", output_tokens)
print("Generation time:", f"{total_time:.2f} sec")

if output_tokens > 0:
    print(
        "Approx. generation speed:",
        f"{output_tokens / total_time:.2f} tokens/sec"
    )

if torch.cuda.is_available():
    print(
        "Peak GPU memory:",
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )

answer = tokenizer.decode(
    outputs[0][input_tokens:],
    skip_special_tokens=True
)

print("\n===== RESPONSE =====")
print(answer)
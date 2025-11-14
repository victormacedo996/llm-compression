from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Path to your model directory
model_dir = "/mnt/3c2f822b-db13-4837-ba6e-3d7b256042cc/repositorios/mestrado/llm-compression/models/models/qwen-compressed/stage3-lora-finetuned"  # Update this path

# Load the base Qwen2 0.6B model
base_model_name = "/mnt/3c2f822b-db13-4837-ba6e-3d7b256042cc/repositorios/mestrado/llm-compression/models/models/qwen-compressed/stage1-quantized"  # or "Qwen/Qwen2-0.5B" depending on your base
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,  # Use float16 for efficiency
    device_map="auto",  # Automatically place on GPU/CPU
)

# Load tokenizer from your fine-tuned directory
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load and merge LoRA adapters
model = PeftModel.from_pretrained(base_model, model_dir, device_map="auto")

# Optional: Merge adapters into base model for faster inference
# model = model.merge_and_unload()

print("Model loaded successfully!")

# ============ INFERENCE EXAMPLES ============


def generate_response(prompt, max_length=128, temperature=0.7, top_p=0.9):
    """Generate text using the model"""

    # Tokenize input
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode and return
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Example 1: Simple text generation
prompt = "The future of AI is"
response = generate_response(prompt)
print(f"Prompt: {prompt}")
print(f"Response: {response}\n")

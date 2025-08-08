
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import oneshot

import argparse 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model using SmoothQuant.")
    parser.add_argument("--recipe", type=str, required=True, help="recipe name or path")
    parser.add_argument("--output", type=str, required=True, help="output name or path")
    args = parser.parse_args()
    # Load model
    model_stub = "Qwen/Qwen2.5-32B"
    model_name = model_stub.split("/")[-1]

    num_samples = 1024
    max_seq_len = 8192

    tokenizer = AutoTokenizer.from_pretrained(model_stub)

    model = AutoModelForCausalLM.from_pretrained(
        model_stub,
        device_map="auto",
        torch_dtype="auto",
    )

    def preprocess_fn(example):
      return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}

    ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
    ds = ds.map(preprocess_fn)

    # Configure the quantization algorithm and scheme
    recipe = args.recipe

    # Apply quantization
    oneshot(
        model=model,
        dataset=ds, 
        recipe=recipe,
        max_seq_length=max_seq_len,
        num_calibration_samples=num_samples,
    )

    # Save to disk in compressed-tensors format
    save_path = model_name + "-quantized.w8a8-" + args.output
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to: {save_path}")
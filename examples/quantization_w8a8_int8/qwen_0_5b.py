
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import oneshot

import argparse 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model using SmoothQuant.")
    parser.add_argument("--input", type=str, required=False, default="/code/model/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=str, required=True, help="output name or path")
    args = parser.parse_args()
    # Load model
    model_stub = args.input
    model_name = model_stub.split("/")[-1]

    tokenizer = AutoTokenizer.from_pretrained(model_stub)
    
    # Configure the simple PTQ quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_stub,
        device_map="cpu",
        torch_dtype="auto",
    )

    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
    )

    # Apply the quantization algorithm.
    oneshot(model=model, recipe=recipe)

    # Save to disk in compressed-tensors format
    save_path = model_name + "-FP8-Dynamic-" + args.output
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to: {save_path}")
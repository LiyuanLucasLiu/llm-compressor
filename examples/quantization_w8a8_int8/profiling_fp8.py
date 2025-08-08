import argparse
from transformers import AutoModelForCausalLM
import torch

def least_square(from_p, to_p, dim=-1):
    to_p = to_p.float()

    beta = (to_p * from_p).sum(dim=dim, keepdim=True) / (from_p **2).sum(dim=dim, keepdim=True)
    
    return beta
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a model with quantization")
    parser.add_argument("--quantized_model", type=str, required=True, help="Path to the quantization config file")
    parser.add_argument("--save_to", type=str, required=True)
    args = parser.parse_args()

    # Load the model and tokenizer
    qmodel = AutoModelForCausalLM.from_pretrained(args.quantized_model)
    
    profile = dict()
    
    qparam = {k: v for k, v in qmodel.named_parameters()}
    
    input_scale = dict()
    for k, v in qparam.items():
        if k.endswith('_scale'):
            profile[k.replace('_scale', '')] = 'minmax'
                    
    torch.save(profile, args.save_to)
    print(f"Profile saved to {args.save_to}")
    
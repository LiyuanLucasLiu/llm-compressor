import argparse
from transformers import AutoModelForCausalLM
import torch

def least_square(from_p, to_p):
    to_p = to_p.float()

    to_pm = to_p - to_p.mean(dim=-1, keepdim=True)
    from_p = from_p - from_p.mean(dim=-1, keepdim=True)

    beta = (to_p * from_p).sum(dim=-1, keepdim=True) / (from_p **2).sum(dim=-1, keepdim=True)
    alpha = (to_p- from_p * beta).mean(dim=-1, keepdim=True)
    
    return beta, alpha
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a model with quantization")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to profile")
    parser.add_argument("--quantized_model", type=str, required=True, help="Path to the quantization config file")
    parser.add_argument("--save_to", type=str, required=True)
    args = parser.parse_args()

    # Load the model and tokenizer
    m = AutoModelForCausalLM.from_pretrained(args.model)
    qmodel = AutoModelForCausalLM.from_pretrained(args.quantized_model)
    
    profile = dict()
    
    qparam = {k: v for k, v in qmodel.named_parameters()}
    for k, v in m.named_parameters():
        if torch.is_floating_point(qparam[k].data) or torch.is_complex(qparam[k].data):
            profile[k] = {
                'beta': 1.0, 
                'alpha': 0.0,
                'type': qparam[k].data.dtype,
            }
        else:
            beta, alpha = least_square(v, qparam[k])
            profile[k] = {
                'beta': beta, 
                'alpha': alpha,
                'type': qparam[k].data.dtype,
            }
    
    torch.save(profile, args.save_to)
    print(f"Profile saved to {args.save_to}")
    
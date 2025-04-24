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
    
    param = {k: v for k, v in m.named_parameters()}
    qparam = {k: v for k, v in qmodel.named_parameters()}
    
    match_dict = {
        'self_attn.q_proj': 'input_layernorm',
        'self_attn.k_proj': 'input_layernorm',
        'self_attn.v_proj': 'input_layernorm',
        'mlp.gate_proj': 'post_attention_layernorm',
        'mlp.up_proj': 'post_attention_layernorm',
    }
    for k, v in param.items():
        if not (torch.is_floating_point(qparam[k].data) or torch.is_complex(qparam[k].data)):
            input_scale = 1.
            for balance, smooth, in match_dict.items():
                if balance in k:
                    layernorm_name = k.replace(balance, smooth)
                    input_scale = param[layernorm_name] / qparam[layernorm_name].view(1, -1)
                    break
        
            profile[k] = {
                'input_scale': input_scale,
                'output_scale': qparam[k+'_scale'].view(-1, 1),
                'type': qparam[k].data.dtype,
            }
        elif 'layernorm' in k: 
            profile[k] = {
                'input_scale': qparam[k] / param[k],
                'output_scale': 1.,
                'type': qparam[k].data.dtype,
            }
    
    torch.save(profile, args.save_to)
    print(f"Profile saved to {args.save_to}")
    
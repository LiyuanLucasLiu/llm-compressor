import argparse
from transformers import AutoModelForCausalLM
import torch

def least_square(from_p, to_p, dim=-1):
    to_p = to_p.float()

    beta = (to_p * from_p).sum(dim=dim, keepdim=True) / (from_p **2).sum(dim=dim, keepdim=True)
    
    return beta
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a model with quantization")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to profile")
    parser.add_argument("--quantized_model", type=str, required=True, help="Path to the quantization config file")
    parser.add_argument("--save_to", type=str, required=True)
    args = parser.parse_args()

    # Load the model and tokenizer
    m = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu")
    qmodel = AutoModelForCausalLM.from_pretrained(args.quantized_model)
    
    profile = dict()
    
    param = {k: v for k, v in m.named_parameters()}
    qparam = {k: v for k, v in qmodel.named_parameters()}
    
    layernorm_list = ['layernorm']
    input_linear_map = {
        'self_attn.q_proj.weight': 'input_layernorm.weight',
        'self_attn.k_proj.weight': 'input_layernorm.weight',
        'self_attn.v_proj.weight': 'input_layernorm.weight',
        'self_attn.o_proj.weight': None,
        'mlp.gate_proj.weight': 'post_attention_layernorm.weight',
        'mlp.up_proj.weight': 'post_attention_layernorm.weight',
    }
    extra_output_list = ['mlp.up_proj.weight']
    output_linear_map = {
        'mlp.down_proj.weight': 'mlp.up_proj.weight',
    }
    exclude_list = [
        'bias', 
        'lm_head.weight',
        'model.norm.weight',
        'embed_tokens',
    ]
    
    input_scale = dict()
    for k, v in param.items():
        for key in layernorm_list:
            if key in k:
                input_scale_k = qparam[k] / param[k].float()
                profile[k] = {
                    'input_scale': input_scale_k,
                    'output_scale': 1.,
                    'type': qparam[k].data.dtype,
                }
                input_scale[k] = input_scale_k 
    
    for k, v in param.items():
        for balance, smooth in input_linear_map.items():
            if balance in k:
                if smooth is None:
                    original_output_scale = qparam[k+'_scale'].view(-1, 1)
                    profile[k] = {
                        'input_scale': 1.0,
                        'output_scale': 1. / original_output_scale.float(),
                        'type': qparam[k].data.dtype,
                    }
                else:
                    input_name = k.replace(balance, smooth)
                    input_scale_k = input_scale[input_name].view(1, -1)
                    original_output_scale = qparam[k+'_scale'].view(-1, 1)
                    
                    if any(ei in k for ei in extra_output_list):
                        additional_output_scale = least_square(
                            (param[k].float() / input_scale_k / original_output_scale).view(param[k].shape[0], -1),
                            qparam[k].view(param[k].shape[0], -1)
                        ).view(-1, 1)
                        input_scale[k] = additional_output_scale
                        original_output_scale = original_output_scale.float() / additional_output_scale
                    profile[k] = {
                        'input_scale': 1. / input_scale_k,
                        'output_scale': 1. / original_output_scale.float(),
                        'type': qparam[k].data.dtype,
                    }
                break
                
    for k, v in param.items():
        for balance, smooth in output_linear_map.items():
            if balance in k:
                input_name = k.replace(balance, smooth)
                input_scale_k = input_scale[input_name].view(1, -1)
                original_output_scale = qparam[k+'_scale'].view(-1, 1)
                profile[k] = {
                    'input_scale': 1. / input_scale_k,
                    'output_scale': 1. / original_output_scale.float(),
                    'type': qparam[k].data.dtype,
                }
                break
    
    for k, v in param.items():
        if k not in profile:
            if not any(ei in k for ei in exclude_list):
                print(k)
                    
    torch.save(profile, args.save_to)
    print(f"Profile saved to {args.save_to}")
    
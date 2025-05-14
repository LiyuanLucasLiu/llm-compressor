import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import shutil

def linear(from_p, profile):
    shifted = from_p * profile['output_scale'] * profile['input_scale']
    if profile['type'] == torch.int8:
        return torch.round(shifted).clamp(min=-128, max=127).to(torch.int8)
    else:
        return shifted.to(profile['type'])
        
input_linear_map = {
    'input_layernorm.weight': ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight'],
    'post_attention_layernorm.weight': ['mlp.gate_proj.weight', 'mlp.up_proj.weight'],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a model with quantization")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to profile")
    parser.add_argument("--quantized_model", type=str, required=True, help="Path to the quantization config file")
    parser.add_argument("--profile", type=str, required=True, help="Path to the quantization config file")
    parser.add_argument("--save_to", type=str, required=True)
    parser.add_argument("--remove_dead", action='store_true', help="Remove dead neurons")
    args = parser.parse_args()

    # Load the model and tokenizer
    m = AutoModelForCausalLM.from_pretrained(args.model, device_map="cpu", torch_dtype="auto")
    qmodel = AutoModelForCausalLM.from_pretrained(args.quantized_model, device_map="cpu", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.quantized_model)

    profile = torch.load(args.profile)
    
    param = {k: v for k, v in m.named_parameters()}
    qparam = {k: v for k, v in qmodel.named_parameters()}
    
    with torch.no_grad():
        # apply the profile to the quantized model
        for k, v in profile.items():
            nd = linear(param[k], v)
            diff = (nd - qparam[k].data).float().norm()
            pn = qparam[k].float().norm()
            print(f'name: {k}, type: {v["type"]}, ratio: {diff / pn}, diff: {diff}, norm: {pn}')
            qparam[k].data = nd
        
        if args.remove_dead:
            # clean dead neurons 
            for k, v in qparam.items():
                for layernorm, layers in input_linear_map.items():
                    if layernorm in k:
                        dead_neurons_mask = 1. 
                        for layer_i in layers:
                            layer_name = k.replace(layernorm, layer_i)
                            dead_neurons_mask_i = (qparam[layer_name].abs().float().sum(dim=0, keepdim=False) == 0)
                            dead_neurons_mask = dead_neurons_mask * dead_neurons_mask_i
                        print(f'layernorm: {k}, dead_neurons_mask ct: {dead_neurons_mask.sum()}')
                        qparam[k].data = qparam[k].data * (1 - dead_neurons_mask)
    
    # Save the quantized model
    qmodel.load_state_dict(qparam, strict=False)
    qmodel.save_pretrained(args.save_to)
    tokenizer.save_pretrained(args.save_to)
    
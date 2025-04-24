import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import shutil

def linear(from_p, profile):
    shifted = from_p * profile['input_scale'] / profile['output_scale']
    if profile['type'] == torch.int8:
        return torch.round(shifted).clamp(min=-128, max=127).to(torch.int8)
    else:
        return shifted.to(profile['type'])
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a model with quantization")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to profile")
    parser.add_argument("--quantized_model", type=str, required=True, help="Path to the quantization config file")
    parser.add_argument("--profile", type=str, required=True, help="Path to the quantization config file")
    parser.add_argument("--save_to", type=str, required=True)
    args = parser.parse_args()

    # Load the model and tokenizer
    m = AutoModelForCausalLM.from_pretrained(args.model)
    qmodel = AutoModelForCausalLM.from_pretrained(args.quantized_model)
    tokenizer = AutoTokenizer.from_pretrained(args.quantized_model)

    profile = torch.load(args.profile)
    
    param = {k: v for k, v in m.named_parameters()}
    qparam = {k: v for k, v in qmodel.named_parameters()}
    
    for k, v in profile.items():
        nd = linear(param[k], v)
        diff = (nd - qparam[k].data).float().norm()
        pn = qparam[k].float().norm()
        print(f'name: {k}, type: {v["type"]}, ratio: {diff / pn}, diff: {diff}, norm: {pn}')
        qparam[k].data = linear(param[k], v)
    
    # Save the quantized model
    qmodel.load_state_dict(qparam)
    qmodel.save_pretrained(args.save_to)
    tokenizer.save_pretrained(args.save_to)
    shutil.copy(args.quantized_model + "/recipe.yaml", args.save_to + "/recipe.yaml")
    shutil.copy(args.quantized_model + "/config.json", args.save_to + "/config.json")
    
import argparse
import json
import os
import time
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset


def load_prompts_from_hf_dataset(dataset_name: str, 
                                 split: str = "train",
                                 text_column: str = "problem",
                                 num_prompts: int = 512,
                                 prompt_template: str = "{text}") -> List[str]:
    """Load prompts from a Hugging Face dataset."""
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)
    
    # Select the specified number of examples (or all if less are available)
    max_examples = min(num_prompts, len(dataset))
    dataset = dataset.select(range(max_examples))
    
    # Extract text from the dataset
    prompts = []
    for example in dataset:
        if text_column in example:
            text = example[text_column]
            # Apply template if needed
            prompt = prompt_template.format(text=text)
            prompts.append(prompt)
        else:
            raise ValueError(f"Column '{text_column}' not found in dataset. Available columns: {list(example.keys())}")
    
    # Warn if we couldn't get the requested number of prompts
    if len(prompts) < num_prompts:
        print(f"Warning: Only {len(prompts)} prompts were available in the dataset (requested {num_prompts})")
    
    return prompts


def run_benchmark(model_name: str,
                  prompts: List[str],
                  output_token_length: int = 128,
                  tensor_parallel_size: int = 1,
                  gpu_memory_utilization: float = 0.95,
                  max_model_len: Optional[int] = None,
                  seed: int = 42,
                  quantization: Optional[str] = None, 
                  batch_size: Optional[int] = None) -> Dict[str, Any]:
    """Run vLLM benchmark and return metrics."""
    
    # Setup vLLM engine
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        seed=seed,
        quantization=quantization,
    )
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for benchmarking
        max_tokens=output_token_length,
    )
    
    print(f"Running benchmark with {len(prompts)} prompts...")
    # Start timing
    start_time = time.time()
    if batch_size is not None:
        # Process prompts in batches
        outputs = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_outputs = llm.generate(batch_prompts, sampling_params)
            outputs.extend(batch_outputs)
    else:
        # Process all prompts at once (vLLM handles batching internally)
        outputs = llm.generate(prompts, sampling_params)
    
    # End timing
    end_time = time.time()
    print(f'Benchmark completed in {end_time - start_time:.2f} seconds')
    
    # Calculate metrics
    elapsed_time = end_time - start_time
    
    # Get input tokens counts
    prompt_token_ids = [llm.get_tokenizer().encode(prompt) for prompt in prompts]
    input_tokens = sum(len(ids) for ids in prompt_token_ids)
    
    # Get output tokens counts
    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    
    # Calculate total tokens (input + output)
    total_tokens = input_tokens + output_tokens
    
    # Calculate throughput metrics
    throughput_tokens_per_second = total_tokens / elapsed_time
    throughput_prompts_per_second = len(prompts) / elapsed_time
    
    # Gather results
    results = {
        "model_name": model_name,
        "num_prompts": len(prompts),
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "elapsed_time_seconds": elapsed_time,
        "tokens_per_second": throughput_tokens_per_second,
        "prompts_per_second": throughput_prompts_per_second,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM's offline batching throughput")
    parser.add_argument("--model", type=str, required=True, 
                        help="Model name or path")
    parser.add_argument("--num-prompts", type=int, default=512,
                        help="Number of prompts to process")
    parser.add_argument("--dataset", type=str, default='inclusionAI/AReaL-RL-Data',
                        help="dataset name or path")
    parser.add_argument("--output-length", type=int, default=128,
                        help="Number of output tokens to generate")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95,
                        help="Fraction of GPU memory to use")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Maximum sequence length")
    parser.add_argument("--output-file", type=str, default="vllm_benchmark_results.json",
                        help="Path to save benchmark results")
    parser.add_argument("--quantization", type=str, default=None,
                        help="Path to save benchmark results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Generate prompts
    prompts = load_prompts_from_hf_dataset(args.dataset, num_prompts=args.num_prompts)
    
    # Print configuration
    print(f"Running benchmark with configuration:")
    print(f"  Model: {args.model}")
    print(f"  Quantization: {args.quantization}")
    print(f"  Number of prompts: {args.num_prompts}")
    print(f"  Output length: {args.output_length} tokens")
    print(f"  Tensor parallel size: {args.tensor_parallel_size}")
    print(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    if args.batch_size is not None:
        print(f"  Batch size: {args.batch_size}")
    
    # Run benchmark
    print("Starting benchmark...")
    results = run_benchmark(
        model_name=args.model,
        prompts=prompts,
        output_token_length=args.output_length,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        quantization=args.quantization,
    )
    
    # Display results
    print("\n######Benchmark Results:")
    print(f"######  Total elapsed time: {results['elapsed_time_seconds']:.2f} seconds")
    print(f"######  Throughput: {results['tokens_per_second']:.2f} tokens/second")
    print(f"######  Prompts per second: {results['prompts_per_second']:.2f}")
    print(f"######  Input tokens: {results['input_tokens']}")
    print(f"######  Output tokens: {results['output_tokens']}")
    print(f"######  Total tokens: {results['total_tokens']}")
    
    # Save results to file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_file}")
    
if __name__ == "__main__":
    main()
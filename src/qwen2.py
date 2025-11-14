import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
from loguru import logger
from core.analysis.profiling.profiler import Profiler
from core.analysis.profiling.models.profiler.llm_profile_options import (
    LLMProfilerOptions,
    AnalyzeConnections,
    MeasureInferenceTime,
    EstimateMemory,
)
import random
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_output(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.7,
        top_p=None,
        do_sample=False,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def compute_neuron_pair_importance(gate_weight, up_weight):
    """
    compute neuron pair importance scores (Maximum Absolute Weight)

    Args:
    - gate_weight: Weight matrix from the gate_proj layer.
    - up_weight: Weight matrix from the up_weight layer.

    Returns:
    - importance_scores: Importance scores for each neuron pair.
    """

    gate_max_abs = torch.max(gate_weight, dim=1).values + torch.abs(
        torch.min(gate_weight, dim=1).values
    )
    up_max_abs = torch.max(up_weight, dim=1).values + torch.abs(
        torch.min(up_weight, dim=1).values
    )
    importance_scores = gate_max_abs + up_max_abs
    return importance_scores


def prune_neuron_pairs(mlp, prune_percent):
    """
    Reduces the dimensions of the **gate_proj**,**up_proj**, **down_proj**
    layers removing the least important neurons.

    Args:
    - mlp: Layers to prune.
    - prune_percent: Percentage of neurons to prune.

    Returns:
    - new_gate_proj, new_up_proj, new_down_proj:  New pruned layers.
    - k: New intermediate size.

    """

    gate_weight = mlp.gate_proj.weight.data.float()
    up_weight = mlp.up_proj.weight.data.float()

    importance_scores = compute_neuron_pair_importance(gate_weight, up_weight)

    original_intermediate_size = gate_weight.size(0)

    num_neuron_pairs_to_prune = min(
        int(prune_percent * original_intermediate_size), original_intermediate_size - 1
    )

    k = original_intermediate_size - num_neuron_pairs_to_prune

    if k <= 0:
        raise ValueError(
            f"Invalid number of neuron pairs to keep: {k}. Adjust the prune_percent."
        )

    _, indices_to_keep = torch.topk(importance_scores, k, largest=True, sorted=True)
    indices_to_keep = indices_to_keep.sort().values

    new_gate_proj = nn.Linear(mlp.gate_proj.in_features, k, bias=False).to(device)
    new_up_proj = nn.Linear(mlp.up_proj.in_features, k, bias=False).to(device)
    new_down_proj = nn.Linear(k, mlp.down_proj.out_features, bias=False).to(device)

    new_gate_proj.weight.data = mlp.gate_proj.weight.data[indices_to_keep, :]
    new_up_proj.weight.data = mlp.up_proj.weight.data[indices_to_keep, :]
    new_down_proj.weight.data = mlp.down_proj.weight.data[:, indices_to_keep]

    return new_gate_proj, new_up_proj, new_down_proj, k


def update_model(model, prune_percent):
    """
    It modifies each mlp layer present in model, to retain only the most
    important neurons. Creating new smaller versions of each layer pruned.

    Args:
    - model: Model to prune.
    - prune_percent: Percentage of neurons to prune.

    Returns:
    - model: New pruned model.
    """
    new_intermediate_size = None

    for idx, layer in enumerate(model.model.layers):

        mlp = layer.mlp

        new_gate_proj, new_up_proj, new_down_proj, new_size = prune_neuron_pairs(
            mlp, prune_percent
        )

        mlp.gate_proj = new_gate_proj
        mlp.up_proj = new_up_proj
        mlp.down_proj = new_down_proj

        if new_intermediate_size is None:
            new_intermediate_size = new_size

    model.config.intermediate_size = new_intermediate_size

    return model


def generate_random_prompt() -> str:
    options = [
        "Explain the theory of relativity in simple terms.",
        "What are the health benefits of a Mediterranean diet?",
        "Describe the process of photosynthesis.",
        "What is the significance of the Renaissance period in history?",
        "How does blockchain technology work?",
        "Translate the following English text to French: 'The quick brown fox jumps over the lazy dog.'",
    ]

    return random.choice(options)


if __name__ == "__main__":

    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm_profiler_opts = LLMProfilerOptions(
        analyze_connections=AnalyzeConnections(
            input_shape=(1, 2048),
        ),
        estimate_memory=EstimateMemory(batch_size=1, sequence_length=100),
        measure_inference_time=MeasureInferenceTime(
            input_sample=generate_random_prompt,
            num_runs=20,
            warmup_runs=2,
            tokenizer_max_length=2048,
        ),
    )
    profiler = Profiler(model, tokenizer, model_name="qwen3-0.6b-uncompressed")
    profiling_results_uncompressed = profiler.profile_complete(llm_profiler_opts)

    with open("./uncompressed_profile_result.json", "w") as file:
        json.dump(
            profiling_results_uncompressed.model_dump(mode="json"), file, indent=4
        )

    logger.info(model)
    prompt = "I am a student that came to you to learn about physics. Can you explain to me the theory of relativity?"
    generated = get_output(prompt, model, tokenizer)
    logger.info(f"Generated text: {generated}")

    original_param_count = count_parameters(model)
    logger.info(f"Original model parameters: {original_param_count}")

    prune_percent = 0.2
    model = update_model(model, prune_percent)
    profiler = Profiler(model, tokenizer, model_name="qwen3-0.6b-pruned")
    profiling_results_pruned = profiler.profile_complete(llm_profiler_opts)

    with open("./pruned_profile_result.json", "w") as file:
        json.dump(profiling_results_pruned.model_dump(mode="json"), file, indent=4)

    pruned_param_count = count_parameters(model)
    reduction_in_params = original_param_count - pruned_param_count
    percentage_savings = (reduction_in_params / original_param_count) * 100

    logger.info(f"Pruned model parameters: {pruned_param_count}")
    logger.info(f"Reduction in parameters: {reduction_in_params}")
    logger.info(f"Percentage of weight savings: {percentage_savings:.2f}%")
    generated = get_output(prompt, model, tokenizer)
    logger.info(f"Generated text after pruning: {generated}")

    logger.info(model)

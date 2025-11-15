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


def compute_head_importance(self_attn, num_attention_heads, head_dim):
    """
    Computes importance scores for each attention head based on the L2 norm of its
    corresponding weights in the output projection layer (o_proj).

    Args:
    - self_attn: The attention block (e.g., model.layers[0].self_attn).
    - num_attention_heads: The total number of query heads.
    - head_dim: The dimension of each attention head.

    Returns:
    - importance_scores: A tensor containing the importance score for each head.
    """
    o_proj_weight = self_attn.o_proj.weight.data.float()

    # The input to o_proj is the concatenated output of all heads.
    # We can view the o_proj weight matrix as being composed of blocks,
    # where each block processes the output of one head.
    # o_proj_weight has shape [hidden_size, num_attention_heads * head_dim]

    importance_scores = torch.zeros(num_attention_heads).to(o_proj_weight.device)

    for i in range(num_attention_heads):
        # Extract the block of weights in o_proj corresponding to the i-th head
        start_idx = i * head_dim
        end_idx = (i + 1) * head_dim

        # The shape of this block is [hidden_size, head_dim]
        head_o_weight_block = o_proj_weight[:, start_idx:end_idx]

        # The importance is the L2 norm of this block
        importance_scores[i] = torch.norm(head_o_weight_block, p=2)

    return importance_scores


def prune_attention_heads(self_attn, prune_percent):
    """
    Reduces attention heads by pruning K/V heads and their associated query heads
    to maintain GQA compatibility.
    """
    config = self_attn.config
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = self_attn.head_dim
    num_query_groups = num_attention_heads // num_key_value_heads

    # Compute importance scores for each individual query head
    head_importance_scores = compute_head_importance(
        self_attn, num_attention_heads, head_dim
    )

    # Aggregate scores into K/V group scores
    group_importance_scores = torch.zeros(num_key_value_heads).to(
        head_importance_scores.device
    )
    for i in range(num_key_value_heads):
        start = i * num_query_groups
        end = (i + 1) * num_query_groups
        group_importance_scores[i] = head_importance_scores[start:end].sum()

    # Determine number of K/V heads to keep
    num_kv_heads_to_prune = int(prune_percent * num_key_value_heads)
    if num_kv_heads_to_prune >= num_key_value_heads:
        num_kv_heads_to_prune = num_key_value_heads - 1
    k_kv_heads = num_key_value_heads - num_kv_heads_to_prune

    # Get indices of K/V heads to keep
    _, kv_indices_to_keep = torch.topk(
        group_importance_scores, k_kv_heads, largest=True, sorted=True
    )
    kv_indices_to_keep = kv_indices_to_keep.sort().values

    # --- Create masks for all three projection layers ---
    q_mask, k_mask, v_mask = [], [], []

    for idx in kv_indices_to_keep:
        # Add the block of query heads for this group
        q_start = idx.item() * num_query_groups * head_dim
        q_end = q_start + num_query_groups * head_dim
        q_mask.extend(range(q_start, q_end))

        # Add the single K/V head for this group
        kv_start = idx.item() * head_dim
        kv_end = kv_start + head_dim
        k_mask.extend(range(kv_start, kv_end))
        v_mask.extend(range(kv_start, kv_end))

    # --- Prune all relevant layers ---
    device, dtype = self_attn.q_proj.weight.device, self_attn.q_proj.weight.dtype

    # Prune Q, K, and V projections
    new_q_proj = nn.Linear(
        config.hidden_size, len(q_mask), bias=config.attention_bias
    ).to(device, dtype)
    new_k_proj = nn.Linear(
        config.hidden_size, len(k_mask), bias=config.attention_bias
    ).to(device, dtype)
    new_v_proj = nn.Linear(
        config.hidden_size, len(v_mask), bias=config.attention_bias
    ).to(device, dtype)

    # Prune output projection (O)
    new_o_proj = nn.Linear(
        len(q_mask), config.hidden_size, bias=config.attention_bias
    ).to(device, dtype)

    # Assign sliced weights
    new_q_proj.weight.data = self_attn.q_proj.weight.data[q_mask, :]
    new_k_proj.weight.data = self_attn.k_proj.weight.data[k_mask, :]
    new_v_proj.weight.data = self_attn.v_proj.weight.data[v_mask, :]
    new_o_proj.weight.data = self_attn.o_proj.weight.data[:, q_mask]

    # Replace layers
    self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj = (
        new_q_proj,
        new_k_proj,
        new_v_proj,
        new_o_proj,
    )

    # Update config
    k_heads = k_kv_heads * num_query_groups
    self_attn.config.num_attention_heads = k_heads
    self_attn.config.num_key_value_heads = k_kv_heads

    return self_attn, k_heads, k_kv_heads


# --- MODIFIED ORCHESTRATOR ---
def update_model_attention(model, num_kv_heads_to_keep: int):
    """
    Orchestrator function that prunes to a specific number of K/V heads.
    """
    # Calculate the prune_percent dynamically based on the target
    original_kv_heads = model.config.num_key_value_heads
    if num_kv_heads_to_keep >= original_kv_heads:
        print("Target K/V heads is >= original. No pruning will be performed.")
        return model

    prune_percent = (original_kv_heads - num_kv_heads_to_keep) / original_kv_heads

    # The rest of the function remains the same
    new_num_attention_heads, new_num_key_value_heads = None, None
    print(
        f"\n--- Starting Attention Head Pruning (Targeting {num_kv_heads_to_keep} K/V heads) ---"
    )

    for idx, layer in enumerate(model.model.layers):
        self_attn = layer.self_attn
        if self_attn.config.num_attention_heads == self_attn.config.num_key_value_heads:
            print(
                f"Skipping attention pruning for layer {idx}: Not GQA or already fully pruned."
            )
            continue
        # Pass the dynamically calculated prune_percent
        _, k_heads, k_kv_heads = prune_attention_heads(self_attn, prune_percent)
        if new_num_attention_heads is None:
            new_num_attention_heads = k_heads
            new_num_key_value_heads = k_kv_heads

    if new_num_attention_heads is not None:
        model.config.num_attention_heads = new_num_attention_heads
        model.config.num_key_value_heads = new_num_key_value_heads
        print(
            f"Attention pruning complete. New heads Q:{new_num_attention_heads}, KV:{new_num_key_value_heads}"
        )
    else:
        print("No attention heads were pruned.")
    return model


# --- HOW TO USE IT ---
# model = update_model_attention(model, num_kv_heads_to_keep=6) # Prunes from 8 to 6 K/V heads (25% reduction)
# model = update_model_attention(model, num_kv_heads_to_keep=4) # Prunes from 8 to 4 K/V heads (50% reduction)


def get_output(prompt, model, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    outputs = model.generate(
        model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        max_new_tokens=100,
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


def update_model_mlp(model, prune_percent):
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

    # prune_percent = 0.2
    # model = update_model_mlp(model, prune_percent)
    # profiler = Profiler(model, tokenizer, model_name="qwen3-0.6b-pruned-mlp")
    # profiling_results_pruned_mlp = profiler.profile_complete(llm_profiler_opts)

    # with open("./pruned_mlp_profile_result.json", "w") as file:
    #     json.dump(profiling_results_pruned_mlp.model_dump(mode="json"), file, indent=4)

    # pruned_param_count = count_parameters(model)
    # reduction_in_params = original_param_count - pruned_param_count
    # percentage_savings = (reduction_in_params / original_param_count) * 100

    model = update_model_attention(model, 7)
    profiler = Profiler(model, tokenizer, model_name="qwen3-0.6b-pruned-attn-n-mlp")
    profiling_results_pruned_mlp_plus_attention_heads = profiler.profile_complete(
        llm_profiler_opts
    )

    with open("./pruned_mlp__plus_attention_heads_profile_result.json", "w") as file:
        json.dump(
            profiling_results_pruned_mlp_plus_attention_heads.model_dump(mode="json"),
            file,
            indent=4,
        )

    generated = get_output(prompt, model, tokenizer)
    logger.info(f"Generated text after pruning: {generated}")

    logger.info(model)

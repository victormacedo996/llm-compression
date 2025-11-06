from core.analysis.profiling.models.llm import (
    AnalyzeConnections,
    EstimateMemory,
    MeasureInferenceTime,
)
from core.analysis.profiling.models.profiler.llm_profile_options import (
    LLMProfilerOptions,
)
import random
from core.orchestration.workflow.compression_workflow import CompressionWorkflow
from core.compression.engine.techniques.quantization.transformers_lib_interface import (
    ITransformersQuantization,
)
from core.compression.engine.techniques.prune.interface import IPrune
from typing import Any, Tuple, Dict
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
import torch.nn as nn
from pathlib import Path
import json


class BitsNBytesQuantizationTechnique(ITransformersQuantization):
    def __init__(
        self, technique_name: str, quantization_args: Dict[str, Any], **kwargs
    ) -> None:
        self.technique_name = technique_name
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.quantization_args = quantization_args

    def quantize(self, hf_model_id: str, *args, **kwargs) -> Tuple[Any, Any]:
        logger.info(f"Loading model: {hf_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
        bnb_config = self.get_quantization_config(**kwargs)
        logger.info("Loading model with BitsAndBytes quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )

        logger.info("Model loaded and quantized successfully")

        logger.info("Quantized model and tokenizer saved")

        return model, tokenizer

    def get_quantization_config(self, *args, **kwargs):
        logger.info("extracting kwargs")
        load_in_4bit = self.quantization_args.get("load_in_4bit", True)
        bnb_4_bit_quant_type = self.quantization_args.get("bnb_4_bit_quant_type", "nf4")
        bnb_4bit_use_double_quant = self.quantization_args.get(
            "bnb_4bit_use_double_quant", True
        )

        logger.info("using following args:")
        logger.info(f"load_in_4bit: {load_in_4bit}")
        logger.info(f"bnb_4_bit_quant_type: {bnb_4_bit_quant_type}")
        logger.info(f"bnb_4bit_use_double_quant: {bnb_4bit_use_double_quant}")

        return BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_4_bit_quant_type,
            bnb_4bit_compute_dtype=self.compute_dtype,
        )


class PruneTechnique(IPrune):

    def __init__(self, technique_name, **kwargs):
        self.technique_name = technique_name

    def structured_pruning_ffn(self, model: Any, pruning_ratio: float = 0.3):
        """Prune entire neurons from FFN layers"""
        logger.info("Starting structured pruning on FFN layers...")

        pruned_count = 0
        total_params = 0
        layers_skipped = 0

        for name, module in model.named_modules():
            if "mlp" in name.lower() or "feed_forward" in name.lower():
                if isinstance(module, nn.Linear):
                    if hasattr(module, "weight") and hasattr(module.weight, "data"):
                        try:
                            weight_data = module.weight.data

                            # Skip quantized layers
                            if weight_data.dtype not in [
                                torch.float32,
                                torch.float16,
                                torch.bfloat16,
                            ]:
                                logger.info(
                                    f"Skipping quantized layer {name} (dtype: {weight_data.dtype})"
                                )
                                layers_skipped += 1
                                continue

                            total_params += weight_data.numel()

                            importance = torch.norm(weight_data, p=2, dim=1)
                            threshold = torch.quantile(importance, pruning_ratio)
                            mask = importance > threshold

                            with torch.no_grad():
                                module.weight.data[~mask] = 0

                            pruned_count += (~mask).sum().item()
                            logger.info(
                                f"Pruned {name}: {(~mask).sum().item()} neurons "
                                f"out of {len(importance)}"
                            )

                        except Exception as e:
                            logger.warning(f"Could not prune layer {name}: {e}")
                            layers_skipped += 1
                            continue

        pruning_percentage = (
            (pruned_count / total_params) * 100 if total_params > 0 else 0
        )
        logger.info(
            f"Total structured pruning: {pruning_percentage:.2f}% of parameters "
            f"({layers_skipped} quantized layers skipped)"
        )

        return model

    def unstructured_pruning_attention(self, model: Any, pruning_ratio: float = 0.15):
        """Prune individual weights from attention layers"""
        logger.info("Starting unstructured pruning on attention layers...")

        pruned_count = 0
        total_params = 0
        layers_skipped = 0

        for name, module in model.named_modules():
            if "self_attn" in name.lower() or "attention" in name.lower():
                if isinstance(module, nn.Linear):
                    try:
                        weight_data = module.weight.data

                        if weight_data.dtype not in [
                            torch.float32,
                            torch.float16,
                            torch.bfloat16,
                            torch.uint8,
                        ]:
                            logger.info(
                                f"Skipping quantized attention layer {name} "
                                f"(dtype: {weight_data.dtype})"
                            )
                            layers_skipped += 1
                            continue

                        total_params += weight_data.numel()
                        importance = torch.abs(weight_data)
                        threshold = torch.quantile(importance.flatten(), pruning_ratio)
                        mask = importance > threshold

                        with torch.no_grad():
                            module.weight.data[~mask] = 0

                        pruned_count += (~mask).sum().item()
                        logger.info(f"Pruned {name}: {(~mask).sum().item()} weights")

                    except Exception as e:
                        logger.warning(f"Could not prune attention layer {name}: {e}")
                        layers_skipped += 1
                        continue

        pruning_percentage = (
            (pruned_count / total_params) * 100 if total_params > 0 else 0
        )
        logger.info(
            f"Total unstructured pruning: {pruning_percentage:.2f}% of parameters "
            f"({layers_skipped} quantized layers skipped)"
        )

        return model

    def prune(self, model: Any, tokenizer: Any):
        """Apply both structured and unstructured pruning"""
        logger.info("Applying combined pruning strategy...")
        logger.info("Note: BitsAndBytes quantized layers will be skipped")

        model = self.structured_pruning_ffn(model=model, pruning_ratio=0.3)
        model = self.unstructured_pruning_attention(model=model, pruning_ratio=0.15)

        return model


model_name = "Qwen/Qwen3-0.6B"


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


quantize = BitsNBytesQuantizationTechnique(
    technique_name="bitsandbytes_quantization",
    quantization_args={
        "load_in_4bit": True,
        "bnb_4_bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    },
)

prune = PruneTechnique(
    technique_name="combined_pruning",
)

compression_workflow = CompressionWorkflow(
    hf_model_id=model_name,
    compression_technique=[prune, quantize],
    checkpoint_dir=Path("./checkpoint"),
)
model_profile = compression_workflow.profile_base_model(llm_profiler_opts)
with open("./pre-compression.json", "w") as file:
    json.dump(model_profile.model_dump_json(), file, indent=4)
compression_workflow.compress_model()
model_profile = compression_workflow.profile_compressed_model(llm_profiler_opts)
with open("./post-compression.json", "w") as file:
    json.dump(model_profile.model_dump_json(), file, indent=4)

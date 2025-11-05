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
from bitsandbytes.nn import Linear4bit, Linear8bitLt


class BitsNBytesQuantizationTechnique(ITransformersQuantization):
    def __init__(
        self, technique_name: str, quantization_args: Dict[str, Any], **kwargs
    ) -> None:
        self.technique_name = technique_name
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.quantization_args = quantization_args

    def quantize(self, hf_model_id: str, *args, **kwargs) -> Tuple[Any, Any]:
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

        logger.info(f"Loading model: {hf_model_id}")

        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_4_bit_quant_type,
            bnb_4bit_compute_dtype=self.compute_dtype,
        )

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


class PruneTechnique(IPrune):

    def __init__(self, technique_name: str, **kwargs):
        self.technique_name = technique_name
        self.prune_ffn_ratio = kwargs.get("prune_ffn_ratio", 0.3)
        self.prune_attention_ratio = kwargs.get("prune_attention_ratio", 0.15)

    def _get_weight_for_pruning(self, module: nn.Module) -> torch.Tensor:
        """Extract weight tensor from both regular and quantized layers"""
        if isinstance(module, (Linear4bit, Linear8bitLt)):
            # For BitsAndBytes quantized layers, dequantize to get original values
            if hasattr(module, "weight"):
                # Dequantize: weight is stored as (data, scale)
                weight = module.weight.data
                if weight.dtype == torch.uint8:
                    # Dequantize using scale factor
                    if hasattr(module.weight, "quant_state"):
                        return module.weight.dequantize()
                    # Fallback: use absolute values of quantized data
                    return weight.float()
                return weight
        elif isinstance(module, nn.Linear):
            return module.weight.data
        elif isinstance(module, nn.Conv2d):
            return module.weight.data
        return None

    def _set_pruned_weight(
        self, module: nn.Module, weight: torch.Tensor, mask: torch.Tensor
    ):
        """Set pruned weights back to the module"""
        # IMPORTANT: modifying `module.weight.data` directly on BitsAndBytes' quantized
        # layers (Linear4bit / Linear8bitLt) will leave the layer's `quant_state`
        # inconsistent with the raw data buffer. That corrupts the internal
        # quantization bookkeeping and leads to runtime errors during forward
        # (e.g. mismatched shapes during dequantize). Until a proper
        # dequantize->prune->requantize flow is implemented, skip in-place
        # pruning for quantized layers to avoid corrupting their internal state.
        if isinstance(module, (Linear4bit, Linear8bitLt)):
            logger.warning(
                "Skipping in-place pruning for BitsAndBytes quantized layer '%s' — re-quantization not implemented.",
                getattr(module, "name", str(type(module))),
            )
            return

        # Regular (non-quantized) layers: apply pruning in-place.
        with torch.no_grad():
            module.weight.data[~mask] = 0

    def structured_pruning_ffn(self, model: Any, prune_ratio: float = 0.3):
        """Prune entire neurons from FFN layers (including quantized)"""
        logger.info("Starting structured pruning on FFN layers...")

        pruned_count = 0
        total_params = 0
        layers_pruned = 0

        for name, module in model.named_modules():
            if "mlp" in name.lower() or "feed_forward" in name.lower():
                if isinstance(module, (nn.Linear, Linear4bit, Linear8bitLt)):
                    try:
                        weight_data = self._get_weight_for_pruning(module)

                        if weight_data is None:
                            continue

                        # Ensure weight is on CPU for computation if needed
                        if weight_data.device.type == "cuda":
                            weight_data = weight_data.cpu()

                        total_params += weight_data.numel()

                        # Calculate importance using L2 norm
                        importance = torch.norm(weight_data.float(), p=2, dim=1)
                        threshold = torch.quantile(importance, prune_ratio)
                        mask = importance > threshold

                        # Apply pruning
                        self._set_pruned_weight(module, weight_data, mask)

                        pruned_neurons = (~mask).sum().item()
                        pruned_count += pruned_neurons
                        layers_pruned += 1

                        logger.info(
                            f"Pruned {name}: {pruned_neurons} neurons "
                            f"out of {len(importance)} "
                            f"(quantized={isinstance(module, (Linear4bit, Linear8bitLt))})"
                        )

                    except Exception as e:
                        logger.warning(f"Could not prune layer {name}: {e}")
                        continue

        pruning_percentage = (
            (pruned_count / total_params) * 100 if total_params > 0 else 0
        )
        logger.info(
            f"Total structured pruning: {pruning_percentage:.2f}% of parameters "
            f"({layers_pruned} layers pruned)"
        )

        return model

    def unstructured_pruning_attention(self, model: Any, prune_ratio: float = 0.15):
        """Prune individual weights from attention layers (including quantized)"""
        logger.info("Starting unstructured pruning on attention layers...")

        pruned_count = 0
        total_params = 0
        layers_pruned = 0

        for name, module in model.named_modules():
            if "self_attn" in name.lower() or "attention" in name.lower():
                if isinstance(module, (nn.Linear, Linear4bit, Linear8bitLt)):
                    try:
                        weight_data = self._get_weight_for_pruning(module)

                        if weight_data is None:
                            continue

                        # Ensure weight is on CPU for computation if needed
                        if weight_data.device.type == "cuda":
                            weight_data = weight_data.cpu()

                        total_params += weight_data.numel()

                        # Calculate importance using absolute values
                        importance = torch.abs(weight_data.float())
                        threshold = torch.quantile(importance.flatten(), prune_ratio)
                        mask = importance > threshold

                        # Apply pruning
                        self._set_pruned_weight(module, weight_data, mask)

                        pruned_weights = (~mask).sum().item()
                        pruned_count += pruned_weights
                        layers_pruned += 1

                        logger.info(
                            f"Pruned {name}: {pruned_weights} weights "
                            f"(quantized={isinstance(module, (Linear4bit, Linear8bitLt))})"
                        )

                    except Exception as e:
                        logger.warning(f"Could not prune attention layer {name}: {e}")
                        continue

        pruning_percentage = (
            (pruned_count / total_params) * 100 if total_params > 0 else 0
        )
        logger.info(
            f"Total unstructured pruning: {pruning_percentage:.2f}% of parameters "
            f"({layers_pruned} layers pruned)"
        )

        return model

    def prune(self, model: Any, tokenizer: Any) -> Any:
        """Apply both structured and unstructured pruning"""
        logger.info("Applying combined pruning strategy...")

        # No BitsAndBytes layers: prune in-place as before
        model = self.structured_pruning_ffn(
            model=model, prune_ratio=self.prune_ffn_ratio
        )
        model = self.unstructured_pruning_attention(
            model=model, prune_ratio=self.prune_attention_ratio
        )

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
        input_shape=(1, 512),
    ),
    estimate_memory=EstimateMemory(batch_size=1, sequence_length=100),
    measure_inference_time=MeasureInferenceTime(
        input_sample=generate_random_prompt,
        num_runs=20,
        warmup_runs=2,
        tokenizer_max_length=512,
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
    prune_ffn_ratio=0.3,
    prune_attention_ratio=0.15,
)


compression_workflow = CompressionWorkflow(model_name, [prune, quantize])
logger.info(compression_workflow.profile_base_model(llm_profiler_opts))
compression_workflow.compress_model()
logger.info(compression_workflow.profile_compressed_model(llm_profiler_opts))

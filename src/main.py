# from core.analysis.profiling.models.llm import (
#     AnalyzeConnections,
#     EstimateMemory,
#     MeasureInferenceTime,
# )
# from core.analysis.profiling.models.profiler.llm_profile_options import (
#     LLMProfilerOptions,
# )
# import random
# from core.orchestration.workflow.compression_workflow import CompressionWorkflow
# from core.compression.engine.techniques.quantization.transformers_lib_interface import (
#     ITransformersQuantization,
# )
# from core.compression.engine.techniques.prune.interface import IPrune
# from typing import Any, Tuple, Dict
# from loguru import logger
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import BitsAndBytesConfig
# import torch
# import torch.nn as nn
# from pathlib import Path
# import json
# import torch.quantization as tq


# class PyTorchQuantizationTechnique(ITransformersQuantization):
#     """
#     Pure PyTorch quantization replacing BitsAndBytes for better inference throughput.
#     Supports both dynamic and static quantization with proper calibration.
#     """

#     def __init__(
#         self,
#         technique_name: str,
#         quantization_args: Dict[str, Any],
#         backend: str = "fbgemm",
#         is_transformers: bool = False,
#         is_pytorch: bool = True,
#         **kwargs
#     ) -> None:
#         self.technique_name = technique_name
#         self.quantization_args = quantization_args
#         self.backend = backend
#         torch.backends.quantized.engine = self.backend
#         self.is_transformers = is_transformers
#         self.is_pytorch = is_pytorch
#         logger.info(f"Using quantization backend: {self.backend}")

#     def quantize(
#         self,
#         model: Any,
#         tokenizer: Any,
#         *args,
#         calibration_data=None,
#         num_calibration_batches: int = 100,
#         **kwargs
#     ) -> Tuple[Any, Any]:
#         """
#         Load and quantize model using pure PyTorch.

#         Args:
#             hf_model_id: Model identifier from HuggingFace
#             calibration_data: DataLoader for static quantization calibration
#             num_calibration_batches: Number of batches for calibration
#         """
#         logger.info("Model loaded. Starting quantization...")

#         # Apply quantization based on scheme
#         scheme = self.quantization_args.get("scheme", "dynamic")

#         if scheme == "dynamic":
#             model = self._apply_dynamic_quantization(model)
#         elif scheme == "static":
#             if calibration_data is None:
#                 logger.warning(
#                     "Static quantization requires calibration data. "
#                     "Falling back to dynamic quantization."
#                 )
#                 model = self._apply_dynamic_quantization(model)
#             else:
#                 model = self._apply_static_quantization(
#                     model, calibration_data, num_calibration_batches
#                 )
#         else:
#             raise ValueError(f"Unknown quantization scheme: {scheme}")

#         # Move to GPU if available
#         if torch.cuda.is_available():
#             model = model.to("cuda")
#             logger.info("Model moved to CUDA")

#         logger.info("Quantization complete")
#         return model, tokenizer

#     def _apply_dynamic_quantization(self, model: nn.Module) -> nn.Module:
#         """
#         Apply dynamic quantization (INT8 weights, FP32 activations).
#         Faster to apply, no calibration needed.
#         """
#         logger.info("Applying dynamic quantization (INT8 weights)...")

#         # FIXED: Only quantize Linear layers
#         # Embeddings cannot use the standard QConfig for activation observers
#         quantized_model = tq.quantize_dynamic(
#             model,
#             {nn.Linear},  # Only Linear layers - this avoids the AssertionError
#             dtype=torch.qint8,
#             inplace=False,
#         )

#         logger.info("Dynamic quantization applied successfully")
#         return quantized_model

#     def _apply_static_quantization(
#         self,
#         model: nn.Module,
#         calibration_data,
#         num_batches: int = 100,
#     ) -> nn.Module:
#         """
#         Apply static quantization (INT8 weights AND activations).
#         Requires calibration but provides better performance.
#         """
#         logger.info("Applying static quantization (INT8 weights + activations)...")

#         # Set default QConfig for Linear layers
#         model.qconfig = tq.get_default_qconfig("fbgemm")
#         logger.info(f"Using QConfig: {model.qconfig}")

#         # CRITICAL FIX: Explicitly disable quantization for embeddings
#         # This prevents the "float_qparams_weight_only_qconfig" assertion error
#         for name, module in model.named_modules():
#             if isinstance(module, nn.Embedding):
#                 module.qconfig = None
#                 logger.info(f"Skipping embedding layer: {name}")

#         # Prepare model for calibration
#         tq.prepare(model, inplace=True)
#         logger.info("Model prepared for calibration")

#         # Run calibration
#         logger.info(f"Calibrating on {num_batches} batches...")
#         model.eval()
#         with torch.no_grad():
#             for i, batch in enumerate(calibration_data):
#                 if i >= num_batches:
#                     break

#                 input_ids = batch[0] if isinstance(batch, (list, tuple)) else batch
#                 input_ids = input_ids.to("cpu")

#                 try:
#                     _ = model(input_ids)
#                 except Exception as e:
#                     logger.warning(f"Calibration batch {i+1} failed: {e}")
#                     continue

#                 if (i + 1) % 10 == 0:
#                     logger.debug(f"Calibrated on batch {i + 1}/{num_batches}")

#         # Convert to quantized model
#         tq.convert(model, inplace=True)
#         logger.info("Static quantization applied successfully")

#         return model

#     def get_quantization_config(self, *args, **kwargs):
#         """Compatibility method for interface."""
#         return self.quantization_args


# class BitsNBytesQuantizationTechnique(ITransformersQuantization):
#     def __init__(
#         self, technique_name: str, quantization_args: Dict[str, Any], **kwargs
#     ) -> None:
#         self.technique_name = technique_name
#         use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
#         self.compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
#         self.quantization_args = quantization_args

#     def quantize(self, hf_model_id: str, *args, **kwargs) -> Tuple[Any, Any]:
#         logger.info(f"Loading model: {hf_model_id}")
#         tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
#         bnb_config = self.get_quantization_config(**kwargs)
#         logger.info("Loading model with BitsAndBytes quantization...")
#         model = AutoModelForCausalLM.from_pretrained(
#             hf_model_id,
#             quantization_config=bnb_config,
#             device_map="auto",
#             trust_remote_code=True,
#             attn_implementation="eager",
#         )

#         logger.info("Model loaded and quantized successfully")

#         logger.info("Quantized model and tokenizer saved")

#         return model, tokenizer

#     def get_quantization_config(self, *args, **kwargs):
#         logger.info("extracting kwargs")
#         load_in_4bit = self.quantization_args.get("load_in_4bit", True)
#         bnb_4_bit_quant_type = self.quantization_args.get("bnb_4_bit_quant_type", "nf4")
#         bnb_4bit_use_double_quant = self.quantization_args.get(
#             "bnb_4bit_use_double_quant", True
#         )

#         logger.info("using following args:")
#         logger.info(f"load_in_4bit: {load_in_4bit}")
#         logger.info(f"bnb_4_bit_quant_type: {bnb_4_bit_quant_type}")
#         logger.info(f"bnb_4bit_use_double_quant: {bnb_4bit_use_double_quant}")

#         return BitsAndBytesConfig(
#             load_in_4bit=load_in_4bit,
#             bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
#             bnb_4bit_quant_type=bnb_4_bit_quant_type,
#             bnb_4bit_compute_dtype=self.compute_dtype,
#         )


# class PruneTechnique(IPrune):

#     def __init__(self, technique_name, **kwargs):
#         self.technique_name = technique_name

#     def structured_pruning_ffn(self, model: Any, pruning_ratio: float = 0.3):
#         """Prune entire neurons from FFN layers"""
#         logger.info("Starting structured pruning on FFN layers...")

#         pruned_count = 0
#         total_params = 0
#         layers_skipped = 0

#         for name, module in model.named_modules():
#             if "mlp" in name.lower() or "feed_forward" in name.lower():
#                 if isinstance(module, nn.Linear):
#                     if hasattr(module, "weight") and hasattr(module.weight, "data"):
#                         try:
#                             weight_data = module.weight.data

#                             # Skip quantized layers
#                             if weight_data.dtype not in [
#                                 torch.float32,
#                                 torch.float16,
#                                 torch.bfloat16,
#                             ]:
#                                 logger.info(
#                                     f"Skipping quantized layer {name} (dtype: {weight_data.dtype})"
#                                 )
#                                 layers_skipped += 1
#                                 continue

#                             total_params += weight_data.numel()

#                             importance = torch.norm(weight_data, p=2, dim=1)
#                             threshold = torch.quantile(importance, pruning_ratio)
#                             mask = importance > threshold

#                             with torch.no_grad():
#                                 module.weight.data[~mask] = 0

#                             pruned_count += (~mask).sum().item()
#                             logger.info(
#                                 f"Pruned {name}: {(~mask).sum().item()} neurons "
#                                 f"out of {len(importance)}"
#                             )

#                         except Exception as e:
#                             logger.warning(f"Could not prune layer {name}: {e}")
#                             layers_skipped += 1
#                             continue

#         pruning_percentage = (
#             (pruned_count / total_params) * 100 if total_params > 0 else 0
#         )
#         logger.info(
#             f"Total structured pruning: {pruning_percentage:.2f}% of parameters "
#             f"({layers_skipped} quantized layers skipped)"
#         )

#         return model

#     def unstructured_pruning_attention(self, model: Any, pruning_ratio: float = 0.15):
#         """Prune individual weights from attention layers"""
#         logger.info("Starting unstructured pruning on attention layers...")

#         pruned_count = 0
#         total_params = 0
#         layers_skipped = 0

#         for name, module in model.named_modules():
#             if "self_attn" in name.lower() or "attention" in name.lower():
#                 if isinstance(module, nn.Linear):
#                     try:
#                         weight_data = module.weight.data

#                         if weight_data.dtype not in [
#                             torch.float32,
#                             torch.float16,
#                             torch.bfloat16,
#                             torch.uint8,
#                         ]:
#                             logger.info(
#                                 f"Skipping quantized attention layer {name} "
#                                 f"(dtype: {weight_data.dtype})"
#                             )
#                             layers_skipped += 1
#                             continue

#                         total_params += weight_data.numel()
#                         importance = torch.abs(weight_data)
#                         threshold = torch.quantile(importance.flatten(), pruning_ratio)
#                         mask = importance > threshold

#                         with torch.no_grad():
#                             module.weight.data[~mask] = 0

#                         pruned_count += (~mask).sum().item()
#                         logger.info(f"Pruned {name}: {(~mask).sum().item()} weights")

#                     except Exception as e:
#                         logger.warning(f"Could not prune attention layer {name}: {e}")
#                         layers_skipped += 1
#                         continue

#         pruning_percentage = (
#             (pruned_count / total_params) * 100 if total_params > 0 else 0
#         )
#         logger.info(
#             f"Total unstructured pruning: {pruning_percentage:.2f}% of parameters "
#             f"({layers_skipped} quantized layers skipped)"
#         )

#         return model

#     def prune(self, model: Any, tokenizer: Any):
#         """Apply both structured and unstructured pruning"""
#         logger.info("Applying combined pruning strategy...")
#         logger.info("Note: BitsAndBytes quantized layers will be skipped")

#         model = self.structured_pruning_ffn(model=model, pruning_ratio=0.3)
#         model = self.unstructured_pruning_attention(model=model, pruning_ratio=0.15)

#         return model

# model_name = "Qwen/Qwen3-0.6B"


# def generate_random_prompt() -> str:
#     options = [
#         "Explain the theory of relativity in simple terms.",
#         "What are the health benefits of a Mediterranean diet?",
#         "Describe the process of photosynthesis.",
#         "What is the significance of the Renaissance period in history?",
#         "How does blockchain technology work?",
#         "Translate the following English text to French: 'The quick brown fox jumps over the lazy dog.'",
#     ]

#     return random.choice(options)


# llm_profiler_opts = LLMProfilerOptions(
#     analyze_connections=AnalyzeConnections(
#         input_shape=(1, 2048),
#     ),
#     estimate_memory=EstimateMemory(batch_size=1, sequence_length=100),
#     measure_inference_time=MeasureInferenceTime(
#         input_sample=generate_random_prompt,
#         num_runs=20,
#         warmup_runs=2,
#         tokenizer_max_length=2048,
#     ),
# )


# # quantize = BitsNBytesQuantizationTechnique(
# #     technique_name="bitsandbytes_quantization",
# #     quantization_args={
# #         "load_in_4bit": True,
# #         "bnb_4_bit_quant_type": "nf4",
# #         "bnb_4bit_use_double_quant": True,
# #     },
# # )

# quantize = PyTorchQuantizationTechnique(
#     technique_name="pytorch_quant",
#     quantization_args={"scheme": "dynamic"},
# )

# prune = PruneTechnique(
#     technique_name="combined_pruning",
# )

# compression_workflow = CompressionWorkflow(
#     hf_model_id=model_name,
#     compression_technique=[prune, quantize],
#     checkpoint_dir=Path("./checkpoint"),
# )
# model_profile = compression_workflow.profile_base_model(llm_profiler_opts)
# with open("./pre-compression.json", "w") as file:
#     json.dump(model_profile.model_dump(mode='json'), file, indent=4)
# compression_workflow.compress_model()
# model_profile = compression_workflow.profile_compressed_model(llm_profiler_opts)
# with open("./post-compression.json", "w") as file:
#     json.dump(model_profile.model_dump(mode='json'), file, indent=4)
# logger.info("saving compressed model")

# inputs = compression_workflow.compressed_tokenizer(
#         "the future of ai is",
#         return_tensors="pt",
#         truncation=True,
#         max_length=2048,
#     ).to("cpu")

#     # Generate with no gradient computation
# with torch.no_grad():
#     output_ids = compression_workflow.compressed_model.generate(
#         inputs["input_ids"],
#         max_length=2048,
#         temperature=0.7,
#         do_sample=True,
#         top_p=0.9,
#         pad_token_id=compression_workflow.compressed_tokenizer.pad_token_id,
#         eos_token_id=compression_workflow.compressed_tokenizer.eos_token_id,
#         num_beams=1,  # Greedy decoding for quantized models
#         early_stopping=True,
#     )


# generated_text = compression_workflow.compressed_tokenizer(
#     output_ids[0],
#     skip_special_tokens=True,
# )

# logger.info(generated_text)

# # def save_quantized_model_hf_compatible(
# #     model,
# #     tokenizer,
# #     save_dir: str,
# # ):
# #     """
# #     Save quantized model by converting to CPU, detaching from quantization,
# #     and using HuggingFace's save_pretrained with safe_serialization.
# #     """

# #     save_path = Path(save_dir)
# #     save_path.mkdir(parents=True, exist_ok=True)

# #     logger.info(f"Preparing quantized model for HF-compatible save...")

# #     try:
# #         # Move model to CPU (important for serialization)
# #         model = model.cpu()

# #         # Try to save with safe_serialization enabled
# #         # This bypasses the tensor pointer inspection that's causing the error
# #         model.save_pretrained(
# #             save_path,
# #             safe_serialization=True,
# #             max_shard_size="2GB",
# #         )
# #         logger.info(f"Model saved with safe_serialization")

# #     except AttributeError as e:
# #         logger.warning(f"HuggingFace save_pretrained failed: {e}")
# #         logger.info("Falling back to torch.save method...")

# #         # Fallback: Save state dict directly
# #         model_checkpoint = {
# #             "state_dict": model.state_dict(),
# #             "model_config": model.config.to_dict() if hasattr(model, "config") else None,
# #         }

# #         model_path = save_path / "pytorch_model.pt"
# #         torch.save(model_checkpoint, model_path)
# #         logger.info(f"Model checkpoint saved to {model_path}")

# #         # Save config
# #         if hasattr(model, "config"):
# #             model.config.save_pretrained(save_path)

# #     # Save tokenizer
# #     tokenizer_path = save_path / "tokenizer"
# #     tokenizer_path.mkdir(parents=True, exist_ok=True)
# #     tokenizer.save_pretrained(tokenizer_path)
# #     logger.info(f"Tokenizer saved successfully")

# #     logger.info(f"Model saved to {save_path}")

# # save_quantized_model_hf_compatible(compression_workflow.compressed_model, compression_workflow.compressed_tokenizer, "/mnt/3c2f822b-db13-4837-ba6e-3d7b256042cc/repositorios/mestrado/llm-compression/checkpoint/Qwen/hf_compatible")

# # import torch
# # from pathlib import Path
# # from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
# # from loguru import logger

# # def load_pytorch_quantized_model(save_dir: str, device: str = "cpu"):
# #     """
# #     Load a PyTorch quantized and pruned model.
# #     Uses torch.jit to load quantized models properly.

# #     Args:
# #         save_dir: Path to saved model directory
# #         device: Device to load on ('cuda' or 'cpu')

# #     Returns:
# #         Tuple of (model, tokenizer)
# #     """
# #     save_path = Path(save_dir)

# #     if not save_path.exists():
# #         raise FileNotFoundError(f"Save directory not found: {save_path}")

# #     logger.info(f"Loading PyTorch quantized model from {save_path}...")

# #     # Load config
# #     config = AutoConfig.from_pretrained(save_path)
# #     logger.info("✓ Config loaded")

# #     # Load tokenizer
# #     tokenizer_path = save_path / "tokenizer"
# #     if tokenizer_path.exists():
# #         tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# #     else:
# #         tokenizer = AutoTokenizer.from_pretrained(save_path)

# #     if tokenizer.pad_token is None:
# #         tokenizer.pad_token = tokenizer.eos_token

# #     logger.info("✓ Tokenizer loaded")

# #     # Load state dict
# #     pt_model_path = save_path / "pytorch_model.pt"
# #     if not pt_model_path.exists():
# #         raise FileNotFoundError(f"pytorch_model.pt not found at {pt_model_path}")

# #     logger.info(f"Loading checkpoint from {pt_model_path}...")
# #     checkpoint = torch.load(pt_model_path, map_location="cpu")

# #     # Extract state dict
# #     if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
# #         state_dict = checkpoint["state_dict"]
# #     else:
# #         state_dict = checkpoint

# #     logger.info(f"State dict contains {len(state_dict)} keys")

# #     # Reconstruct quantized model
# #     # Step 1: Create base model from config
# #     from transformers import AutoModelForCausalLM

# #     model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
# #     logger.info("✓ Base model created from config")

# #     # Step 2: Load the quantized state dict
# #     # This is the critical part - PyTorch quantized modules have special structure
# #     try:
# #         # Try strict loading first
# #         model.load_state_dict(state_dict, strict=True)
# #         logger.info("✓ State dict loaded (strict=True)")

# #     except RuntimeError as e:
# #         logger.warning(f"Strict loading failed: {e}")
# #         logger.info("Attempting flexible loading with strict=False...")

# #         # Use strict=False to handle pruned/quantized layer structure mismatch
# #         incompatible_keys = model.load_state_dict(state_dict, strict=False)

# #         if incompatible_keys.missing_keys:
# #             logger.warning(f"Missing keys ({len(incompatible_keys.missing_keys)}):")
# #             for key in incompatible_keys.missing_keys[:5]:
# #                 logger.warning(f"  - {key}")
# #             if len(incompatible_keys.missing_keys) > 5:
# #                 logger.warning(f"  ... and {len(incompatible_keys.missing_keys) - 5} more")

# #         if incompatible_keys.unexpected_keys:
# #             logger.warning(f"Unexpected keys ({len(incompatible_keys.unexpected_keys)}):")
# #             for key in incompatible_keys.unexpected_keys[:5]:
# #                 logger.warning(f"  - {key}")
# #             if len(incompatible_keys.unexpected_keys) > 5:
# #                 logger.warning(f"  ... and {len(incompatible_keys.unexpected_keys) - 5} more")

# #         logger.info("✓ State dict loaded (strict=False)")

# #     # Step 3: Move to device
# #     model = model.to(device)
# #     logger.info(f"✓ Model moved to {device}")

# #     # Step 4: Set to eval mode
# #     model.eval()
# #     logger.info("✓ Model set to eval mode")

# #     return model, tokenizer


# # def generate_text(
# #     model,
# #     tokenizer,
# #     prompt: str,
# #     max_length: int = 128,
# #     temperature: float = 0.7,
# #     top_p: float = 0.9,
# #     device: str = "cpu",
# # ) -> str:
# #     """
# #     Generate text using the quantized model.

# #     Args:
# #         model: Loaded quantized model
# #         tokenizer: Loaded tokenizer
# #         prompt: Input text prompt
# #         max_length: Maximum length of generated text
# #         temperature: Controls randomness (0.7 is balanced)
# #         top_p: Nucleus sampling (0.9 is recommended)
# #         device: Device model is on

# #     Returns:
# #         Generated text
# #     """
# #     # Tokenize input
# #     inputs = tokenizer(
# #         prompt,
# #         return_tensors="pt",
# #         truncation=True,
# #         max_length=2048,
# #     ).to(device)

# #     # Generate with no gradient computation
# #     with torch.no_grad():
# #         output_ids = model.generate(
# #             inputs["input_ids"],
# #             max_length=max_length,
# #             temperature=temperature,
# #             top_p=top_p,
# #             do_sample=True,
# #             pad_token_id=tokenizer.pad_token_id,
# #             eos_token_id=tokenizer.eos_token_id,
# #         )

# #     # Decode output
# #     generated_text = tokenizer.decode(
# #         output_ids[0],
# #         skip_special_tokens=True,
# #     )

# #     return generated_text


# # def benchmark_model(model, tokenizer, prompt: str, num_iterations: int = 5):
# #     """
# #     Benchmark inference speed of quantized model.
# #     """
# #     import time

# #     logger.info(f"Benchmarking model with {num_iterations} iterations...")

# #     times = []

# #     # Warmup
# #     generate_text(model, tokenizer, prompt, max_length=50)

# #     # Actual benchmark
# #     for i in range(num_iterations):
# #         start = time.time()
# #         generate_text(model, tokenizer, prompt, max_length=100)
# #         elapsed = time.time() - start
# #         times.append(elapsed)
# #         logger.info(f"  Iteration {i+1}: {elapsed:.3f}s")

# #     avg_time = sum(times) / len(times)
# #     logger.info(f"\nBenchmark Results:")
# #     logger.info(f"  Average time: {avg_time:.3f}s")
# #     logger.info(f"  Min time: {min(times):.3f}s")
# #     logger.info(f"  Max time: {max(times):.3f}s")

# #     return avg_time


# # # ============= USAGE EXAMPLE =============

# # if __name__ == "__main__":

# #     save_dir = "/mnt/3c2f822b-db13-4837-ba6e-3d7b256042cc/repositorios/mestrado/llm-compression/checkpoint/Qwen/hf_compatible"

# #     try:
# #         # Load model
# #         logger.info("="*60)
# #         logger.info("Loading quantized model...")
# #         logger.info("="*60)
# #         model, tokenizer = load_pytorch_quantized_model(save_dir, device="cpu")

# #         # Generate text
# #         logger.info("\n" + "="*60)
# #         logger.info("Testing text generation...")
# #         logger.info("="*60)
# #         prompt = "The future of artificial intelligence is"
# #         result = generate_text(
# #             model=model,
# #             tokenizer=tokenizer,
# #             prompt=prompt,
# #             max_length=256,
# #             temperature=0.8,
# #             top_p=0.95,
# #             device="cpu",
# #         )

# #         logger.info(f"\nPrompt: {prompt}")
# #         logger.info(f"Generated: {result}")

# #         # Benchmark
# #         logger.info("\n" + "="*60)
# #         logger.info("Benchmarking inference...")
# #         logger.info("="*60)
# #         avg_time = benchmark_model(model, tokenizer, prompt, num_iterations=5)

# #     except Exception as e:
# #         logger.error(f"Error: {e}", exc_info=True)

import time
from typing import Tuple, Any, Optional, Callable
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from core.analysis.profiling.models.llm import (
    ArchitectureInfo,
    ModelSummary,
    InferenceTimeInfo,
    LLMInfo,
    ParameterInfo,
    AttentionLayerAnalysisInfo,
)


class LLMProfiler:
    def __init__(
        self, model: nn.Module, model_name: str = "Unknown Model", verbose: bool = True
    ):
        self.model = model
        self.model_name = model_name
        self.verbose = verbose

        first_param = next(iter(model.parameters()), None)
        if first_param is not None:
            self.device = first_param.device
        else:
            first_buffer = next(iter(model.buffers()), None)
            self.device = (
                first_buffer.device if first_buffer is not None else torch.device("cpu")
            )

        self.profile_data: LLMInfo | None = None

    def count_parameters(self) -> ParameterInfo:
        total = 0
        trainable = 0
        non_trainable = 0
        dtype_counts = defaultdict(int)
        dtype_bytes = defaultdict(int)

        for p in self.model.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
            else:
                non_trainable += n

            dtype = str(p.dtype)
            dtype_counts[dtype] += n
            dtype_bytes[dtype] += n * p.element_size()

        return ParameterInfo(
            total=total,
            trainable=trainable,
            non_trainable=non_trainable,
            total_millions=round(total / 1e6, 2),
            total_billions=round(total / 1e9, 3),
            by_dtype_counts=dict(dtype_counts),
            by_dtype_bytes={k: int(v) for k, v in dtype_bytes.items()},
        )

    def analyze_architecture(self) -> ArchitectureInfo:
        layer_types = defaultdict(int)
        layer_details = []
        total_layers = 0

        for name, module in self.model.named_modules():
            children = list(module.children())
            if len(children) == 0:  # leaf module
                layer_type = type(module).__name__
                layer_types[layer_type] += 1

                layer_params = sum(p.numel() for p in module.parameters())
                depth = name.count(".") if name else 0

                layer_info = {
                    "name": name if name else type(module).__name__,
                    "type": layer_type,
                    "parameters": layer_params,
                    "depth": depth,
                }

                if hasattr(module, "in_features") and hasattr(module, "out_features"):
                    layer_info["input_size"] = getattr(module, "in_features", None)
                    layer_info["output_size"] = getattr(module, "out_features", None)
                if hasattr(module, "num_embeddings") and hasattr(
                    module, "embedding_dim"
                ):
                    layer_info["vocab_size"] = getattr(module, "num_embeddings", None)
                    layer_info["embedding_dim"] = getattr(module, "embedding_dim", None)
                if hasattr(module, "num_heads"):
                    layer_info["num_heads"] = getattr(module, "num_heads", None)
                if hasattr(module, "num_attention_heads"):
                    layer_info["num_attention_heads"] = getattr(
                        module, "num_attention_heads", None
                    )

                layer_details.append(layer_info)
                total_layers += 1

        max_depth = max((ld["depth"] for ld in layer_details), default=0)
        return ArchitectureInfo(
            total_layers=total_layers,
            layer_types_count=dict(layer_types),
            layer_details=layer_details,
            max_depth=max_depth,
        )

    def measure_inference_time(
        self,
        input_shape: Optional[Tuple[int, ...]] = None,
        input_sample: Optional[Callable[[], Any]] = None,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> InferenceTimeInfo:

        self.model.eval()

        def _prepare_input():
            if input_sample is not None:
                inp = input_sample()
                if isinstance(inp, dict):
                    return {
                        k: (v.to(self.device) if torch.is_tensor(v) else v)
                        for k, v in inp.items()
                    }
                if isinstance(inp, (tuple, list)):
                    return tuple(
                        (v.to(self.device) if torch.is_tensor(v) else v) for v in inp
                    )
                if torch.is_tensor(inp):
                    return inp.to(self.device)
                return inp

            if input_shape is None:
                raise ValueError(
                    "Either input_shape or input_sample must be provided for inference timing."
                )

            if len(input_shape) == 2:
                batch, seq = input_shape
                return torch.randint(
                    0, 1000, (batch, seq), dtype=torch.long, device=self.device
                )
            else:
                return torch.randn(input_shape, device=self.device)

        sample = _prepare_input()

        with torch.no_grad():
            for _ in range(warmup_runs):
                if isinstance(sample, dict):
                    _ = self.model(**sample)
                elif isinstance(sample, (tuple, list)):
                    _ = self.model(*sample)
                else:
                    _ = self.model(sample)

        times = []
        with torch.no_grad():
            if getattr(self.device, "type", "") == "cuda":
                for _ in range(num_runs):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()

                    if isinstance(sample, dict):
                        _ = self.model(**sample)
                    elif isinstance(sample, (tuple, list)):
                        _ = self.model(*sample)
                    else:
                        _ = self.model(sample)

                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed_ms = start_event.elapsed_time(end_event)  # milliseconds
                    times.append(elapsed_ms / 1000.0)  # convert to seconds
            else:
                for _ in range(num_runs):
                    t0 = time.time()
                    if isinstance(sample, dict):
                        _ = self.model(**sample)
                    elif isinstance(sample, (tuple, list)):
                        _ = self.model(*sample)
                    else:
                        _ = self.model(sample)
                    t1 = time.time()
                    times.append(t1 - t0)

        times = np.array(times)
        return InferenceTimeInfo(
            mean_time_ms=round(float(np.mean(times) * 1000), 4),
            std_time_ms=round(float(np.std(times) * 1000), 4),
            min_time_ms=round(float(np.min(times) * 1000), 4),
            max_time_ms=round(float(np.max(times) * 1000), 4),
            median_time_ms=round(float(np.median(times) * 1000), 4),
            runs=int(len(times)),
        )

    def analyze_attention_layers(self) -> AttentionLayerAnalysisInfo:
        attention_candidates = {}  # name -> module

        def looks_like_attention(mod):
            tname = type(mod).__name__.lower()
            if isinstance(mod, torch.nn.MultiheadAttention):
                return True
            if any(
                hasattr(mod, attr)
                for attr in (
                    "num_attention_heads",
                    "num_heads",
                    "all_head_size",
                    "head_dim",
                )
            ):
                return True
            if (
                "attention" in tname
                or "attn" in tname
                or "selfattn" in tname
                or "self_attention" in tname
                or "multihead" in tname
            ):
                return True
            return False

        for name, module in self.model.named_modules():
            if looks_like_attention(module):
                candidate_name = name if name else type(module).__name__
                attention_candidates[candidate_name] = module

        candidate_names = list(attention_candidates.keys())
        leaf_candidate_names = []
        for n in candidate_names:
            is_parent = any(
                (other != n) and other.startswith(n + ".") for other in candidate_names
            )
            if not is_parent:
                leaf_candidate_names.append(n)

        attention_layers = []
        for name in sorted(leaf_candidate_names):
            module = attention_candidates[name]
            info = {"name": name, "type": type(module).__name__}
            if hasattr(module, "num_attention_heads"):
                info["num_attention_heads"] = getattr(module, "num_attention_heads")
            if hasattr(module, "num_heads"):
                info["num_heads"] = getattr(module, "num_heads")
            if hasattr(module, "embed_dim"):
                info["embed_dim"] = getattr(module, "embed_dim")
            if hasattr(module, "head_dim"):
                info["head_dim"] = getattr(module, "head_dim")
            attention_layers.append(info)

        return AttentionLayerAnalysisInfo(
            num_attention_layers=len(attention_layers),
            attention_layers=attention_layers,
        )

    def get_model_summary(self) -> ModelSummary:
        return ModelSummary(
            model_name=self.model_name,
            device=str(self.device),
            model_class=type(self.model).__name__,
            pytorch_version=torch.__version__,
        )

    def profile_complete(
        self,
        input_shape: Optional[Tuple[int, ...]] = None,
        input_sample: Optional[Callable[[], Any]] = None,
        measure_inference: bool = False,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> LLMInfo:
        if self.verbose:
            logger.info(f"🔍 Profiling {self.model_name}...")

        model_summary = self.get_model_summary()
        if self.verbose:
            logger.info("📊 Counting parameters...")
        parameters_info = self.count_parameters()
        if self.verbose:
            logger.info("🏗️  Analyzing architecture...")
        architecture_info = self.analyze_architecture()
        if self.verbose:
            logger.info("🎯 Analyzing attention layers...")
        attention_layer_info = self.analyze_attention_layers()

        inference_time_info = None
        if measure_inference:
            if self.verbose:
                logger.info("⏱️  Measuring inference time...")
            inference_time_info = self.measure_inference_time(
                input_shape=input_shape,
                input_sample=input_sample,
                num_runs=num_runs,
                warmup_runs=warmup_runs,
            )

        llm_info = LLMInfo(
            architecture=architecture_info,
            attention_layers=attention_layer_info,
            inference_time=inference_time_info,
            parameters=parameters_info,
            summary=model_summary,
        )

        self.profile_data = llm_info
        if self.verbose:
            logger.info("✅ Profiling complete!")
        return llm_info

    def print_summary(self):
        if not self.profile_data:
            logger.info("❌ No profiling data available. Run profile_complete() first.")
            return

        logger.info(f"{'='*60}")
        logger.info(f"🤖 MODEL PROFILE SUMMARY: {self.profile_data.summary.model_name}")
        logger.info(f"{'='*60}")

        summary = self.profile_data.summary
        logger.info(f"📋 Model Class: {summary.model_class}")
        logger.info(f"🖥️  Device: {summary.device}")
        logger.info(f"🔧 PyTorch Version: {summary.pytorch_version}")

        params = self.profile_data.parameters
        logger.info("📊 PARAMETERS:")
        logger.info(
            f"   Total: {params.total:,} ({params.total_millions}M / {params.total_billions}B)"
        )
        logger.info(f"   Trainable: {params.trainable:,}")
        logger.info(f"   Non-trainable: {params.non_trainable:,}")
        logger.info(f"   By dtype: {params.by_dtype_counts}")

        arch = self.profile_data.architecture
        logger.info("🏗️  ARCHITECTURE:")
        logger.info(f"   Total Leaf Modules: {arch.total_layers}")
        logger.info(f"   Max Depth (approx): {arch.max_depth}")
        logger.info(f"   Layer Types: {arch.layer_types_count}")

        # memory = self.profile_data['memory']
        # logger.info(f"💾 MEMORY USAGE:")
        # logger.info(f"   Model Size: {memory['model_size_mb']} MB ({memory['model_size_gb']} GB)")
        # if memory['gpu_memory_info']:
        #     gpu_info = memory['gpu_memory_info']
        #     if gpu_info.get('allocated_memory_mb') is not None:
        #         logger.info(f"   GPU Allocated: {gpu_info['allocated_memory_mb']:.2f} MB")
        #     if gpu_info.get('cached_memory_mb') is not None:
        #         logger.info(f"   GPU Cached/Reserved: {gpu_info['cached_memory_mb']:.2f} MB")

        attention = self.profile_data.attention_layers
        logger.info("🎯 ATTENTION LAYERS:")
        logger.info(f"   Number of Attention Layers: {attention.num_attention_layers}")

        if self.profile_data.inference_time:
            timing = self.profile_data.inference_time
            logger.info("⏱️  INFERENCE TIMING:")
            logger.info(f"   Mean: {timing.mean_time_ms} ms (runs={timing.runs})")
            logger.info(f"   Std: {timing.std_time_ms} ms")
            logger.info(f"   Range: {timing.min_time_ms} - {timing.max_time_ms} ms")

        logger.info(f"{'='*60}")

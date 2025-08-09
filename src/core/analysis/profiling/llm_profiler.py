import time
from typing import Tuple, Any, Optional, Callable, Dict
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
    MemoryEstimate,
    MemoryEstimationInfo,
    PrecisionType,
)


class LLMProfiler:
    """Enhanced LLM profiler with memory estimation capabilities."""

    # Precision to bytes mapping
    PRECISION_BYTES = {
        PrecisionType.FP32: 4.0,
        PrecisionType.FP16: 2.0,
        PrecisionType.BFLOAT16: 2.0,
        PrecisionType.INT8: 1.0,
        PrecisionType.INT4: 0.5,
    }

    def __init__(
        self, model: nn.Module, model_name: str = "Unknown Model", verbose: bool = True
    ):
        self.model = model
        self.model_name = model_name
        self.verbose = verbose
        self.device = self._detect_device()
        self.profile_data: Optional[LLMInfo] = None

    def _detect_device(self) -> torch.device:
        """Detect the device where the model is located."""
        # Check parameters first
        first_param = next(iter(self.model.parameters()), None)
        if first_param is not None:
            return first_param.device

        # Fallback to buffers
        first_buffer = next(iter(self.model.buffers()), None)
        if first_buffer is not None:
            return first_buffer.device

        # Default to CPU
        return torch.device("cpu")

    def count_parameters(self) -> ParameterInfo:
        """Count and categorize model parameters."""
        total = 0
        trainable = 0
        non_trainable = 0
        dtype_counts = defaultdict(int)
        dtype_bytes = defaultdict(int)

        for param in self.model.parameters():
            num_params = param.numel()
            total += num_params

            if param.requires_grad:
                trainable += num_params
            else:
                non_trainable += num_params

            dtype_str = str(param.dtype)
            dtype_counts[dtype_str] += num_params
            dtype_bytes[dtype_str] += num_params * param.element_size()

        return ParameterInfo(
            total=total,
            trainable=trainable,
            non_trainable=non_trainable,
            total_millions=round(total / 1e6, 2),
            total_billions=round(total / 1e9, 3),
            by_dtype_counts=dict(dtype_counts),
            by_dtype_bytes={k: int(v) for k, v in dtype_bytes.items()},
        )

    def estimate_memory_requirements(
        self,
        sequence_length: int = 2048,
        batch_size: int = 1,
        include_kv_cache: bool = True,
        include_activations: bool = True,
        include_training: bool = True,
        gradient_accumulation_steps: int = 1,
        optimizer_type: str = "adamw",  # "adamw", "sgd", "adafactor"
    ) -> MemoryEstimationInfo:
        if self.verbose:
            logger.info("💾 Estimating memory requirements...")

        # Get parameter count
        param_info = self.count_parameters()
        total_params = param_info.total

        # Analyze architecture for attention info
        arch_info = self.analyze_architecture()
        attention_info = self.analyze_attention_layers()

        estimates = {}

        for precision_type in PrecisionType:
            bytes_per_param = self.PRECISION_BYTES[precision_type]

            # Model weights memory
            model_weights_mb = (total_params * bytes_per_param) / (1024**2)

            # Base memory (just model weights)
            total_memory_mb = model_weights_mb

            # KV Cache estimation (for inference)
            kv_cache_mb = None
            if (
                include_kv_cache
                and attention_info.num_attention_layers > 0
                and not include_training
            ):
                kv_cache_mb = self._estimate_kv_cache_memory(
                    attention_info, sequence_length, batch_size, bytes_per_param
                )
                total_memory_mb += kv_cache_mb

            # Activation memory estimation
            activation_memory_mb = None
            if include_activations:
                activation_memory_mb = self._estimate_activation_memory(
                    arch_info, sequence_length, batch_size, bytes_per_param
                )
                total_memory_mb += activation_memory_mb

            # Training memory estimation
            training_memory_mb = None
            optimizer_memory_mb = None
            gradient_memory_mb = None

            if include_training:
                training_estimates = self._estimate_training_memory(
                    total_params,
                    bytes_per_param,
                    optimizer_type,
                    gradient_accumulation_steps,
                    precision_type,
                )

                gradient_memory_mb = training_estimates["gradients"]
                optimizer_memory_mb = training_estimates["optimizer"]
                training_memory_mb = training_estimates["total"]

                total_memory_mb += training_memory_mb

            # Total memory with overhead
            overhead_multiplier = (
                1.3 if include_training else 1.2
            )  # Higher overhead for training
            total_with_overhead_mb = total_memory_mb * overhead_multiplier

            logger.debug(training_estimates)

            estimates[precision_type.value] = MemoryEstimate(
                precision=precision_type.value,
                bytes_per_parameter=bytes_per_param,
                total_memory_mb=round(total_memory_mb, 2),
                total_memory_gb=round(total_memory_mb / 1024, 3),
                model_weights_mb=round(model_weights_mb, 2),
                kv_cache_mb=round(kv_cache_mb, 2) if kv_cache_mb else None,
                activation_memory_mb=(
                    round(activation_memory_mb, 2) if activation_memory_mb else None
                ),
                total_inference_memory_mb=(
                    round(total_with_overhead_mb, 2) if not include_training else None
                ),
                # Training-specific fields
                gradient_memory_mb=(
                    round(gradient_memory_mb, 2) if gradient_memory_mb else None
                ),
                optimizer_memory_mb=(
                    round(optimizer_memory_mb, 2) if optimizer_memory_mb else None
                ),
                training_memory_mb=(
                    round(training_memory_mb, 2) if training_memory_mb else None
                ),
                total_training_memory_mb=(
                    round(total_with_overhead_mb, 2) if include_training else None
                ),
            )

        return MemoryEstimationInfo(estimates=estimates, base_parameters=total_params)

    def _estimate_training_memory(
        self,
        total_params: int,
        bytes_per_param: int,
        optimizer_type: str,
        gradient_accumulation_steps: int,
        precision_type: PrecisionType,
    ) -> dict:
        """Estimate memory requirements for training."""

        # Gradient memory (same precision as model)
        gradient_memory_mb = (total_params * bytes_per_param) / (1024**2)

        # Optimizer state memory
        optimizer_memory_mb = 0

        if optimizer_type.lower() == "adamw":
            # AdamW stores: momentum (fp32) + variance (fp32) = 8 bytes per param
            optimizer_memory_mb = (total_params * 8) / (1024**2)
        elif optimizer_type.lower() == "sgd":
            # SGD with momentum stores: momentum (same precision as model)
            optimizer_memory_mb = (total_params * bytes_per_param) / (1024**2)
        elif optimizer_type.lower() == "adafactor":
            # Adafactor is more memory efficient, roughly 4 bytes per param
            optimizer_memory_mb = (total_params * 4) / (1024**2)
        else:
            # Default to AdamW estimation
            optimizer_memory_mb = (total_params * 8) / (1024**2)

        # Gradient accumulation factor
        if gradient_accumulation_steps > 1:
            # Need to store accumulated gradients
            gradient_memory_mb *= gradient_accumulation_steps

        # Mixed precision adjustments
        if (
            precision_type == PrecisionType.FP16
            or precision_type == PrecisionType.BFLOAT16
        ):
            # Master weights in FP32 for mixed precision training
            master_weights_mb = (total_params * 4) / (1024**2)  # FP32
            optimizer_memory_mb += master_weights_mb

        total_training_memory_mb = gradient_memory_mb + optimizer_memory_mb

        return {
            "gradients": gradient_memory_mb,
            "optimizer": optimizer_memory_mb,
            "total": total_training_memory_mb,
        }

    def _estimate_kv_cache_memory(
        self,
        attention_info: AttentionLayerAnalysisInfo,
        sequence_length: int,
        batch_size: int,
        bytes_per_param: float,
    ) -> float:
        """Estimate KV cache memory requirements."""
        total_kv_memory = 0

        for layer in attention_info.attention_layers:
            # Try to extract dimensions from layer info
            num_heads = (
                layer.get("num_attention_heads")
                or layer.get("num_heads")
                or 12  # default fallback
            )

            head_dim = layer.get("head_dim")
            if not head_dim:
                embed_dim = layer.get("embed_dim", 768)  # default fallback
                head_dim = embed_dim // num_heads

            # KV cache: 2 (K and V) * batch_size * num_heads * sequence_length * head_dim
            layer_kv_memory = (
                2
                * batch_size
                * num_heads
                * sequence_length
                * head_dim
                * bytes_per_param
            )
            total_kv_memory += layer_kv_memory

        return total_kv_memory / (1024**2)  # Convert to MB

    def _estimate_activation_memory(
        self,
        arch_info: ArchitectureInfo,
        sequence_length: int,
        batch_size: int,
        bytes_per_param: float,
    ) -> float:
        """Estimate activation memory requirements (rough approximation)."""

        # Find embedding dimension from layer details
        embed_dim = 768  # default fallback
        for layer in arch_info.layer_details:
            if layer.get("embedding_dim"):
                embed_dim = layer["embedding_dim"]
                break
            elif layer.get("output_size"):
                embed_dim = max(embed_dim, layer["output_size"])

        # Rough estimation: batch_size * sequence_length * embed_dim * num_layers * multiplier
        num_layers = len(
            [
                layer
                for layer in arch_info.layer_details
                if "attention" in layer.get("type", "").lower()
            ]
        )
        if num_layers == 0:
            num_layers = 12  # default fallback

        # Activation memory multiplier (accounts for intermediate representations)
        activation_multiplier = 4

        activation_memory_bytes = (
            batch_size
            * sequence_length
            * embed_dim
            * num_layers
            * activation_multiplier
            * bytes_per_param
        )

        return activation_memory_bytes / (1024**2)  # Convert to MB

    def analyze_architecture(self) -> ArchitectureInfo:
        """Analyze model architecture and extract layer information."""
        layer_types = defaultdict(int)
        layer_details = []
        total_layers = 0

        for name, module in self.model.named_modules():
            if not list(module.children()):  # leaf module
                layer_type = type(module).__name__
                layer_types[layer_type] += 1

                layer_params = sum(p.numel() for p in module.parameters())
                depth = name.count(".") if name else 0

                layer_info = {
                    "name": name if name else layer_type,
                    "type": layer_type,
                    "parameters": layer_params,
                    "depth": depth,
                }

                # Extract layer-specific information
                self._extract_layer_attributes(module, layer_info)

                layer_details.append(layer_info)
                total_layers += 1

        max_depth = max((ld["depth"] for ld in layer_details), default=0)

        return ArchitectureInfo(
            total_layers=total_layers,
            layer_types_count=dict(layer_types),
            layer_details=layer_details,
            max_depth=max_depth,
        )

    def _extract_layer_attributes(self, module: nn.Module, layer_info: dict) -> None:
        """Extract relevant attributes from a layer module."""
        # Linear layer attributes
        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            layer_info["input_size"] = getattr(module, "in_features", None)
            layer_info["output_size"] = getattr(module, "out_features", None)

        # Embedding layer attributes
        if hasattr(module, "num_embeddings") and hasattr(module, "embedding_dim"):
            layer_info["vocab_size"] = getattr(module, "num_embeddings", None)
            layer_info["embedding_dim"] = getattr(module, "embedding_dim", None)

        # Attention layer attributes
        for attr in ["num_heads", "num_attention_heads", "head_dim", "embed_dim"]:
            if hasattr(module, attr):
                layer_info[attr] = getattr(module, attr, None)

    def measure_inference_time(
        self,
        input_shape: Optional[Tuple[int, ...]] = None,
        input_sample: Optional[Callable[[], Any]] = None,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> InferenceTimeInfo:
        """Measure model inference time with proper warmup."""
        self.model.eval()

        # Prepare input
        sample = self._prepare_inference_input(input_shape, input_sample)

        # Warmup
        self._run_warmup(sample, warmup_runs)

        # Measure timing
        times = self._measure_timing(sample, num_runs)

        return self._compute_timing_statistics(times)

    def _prepare_inference_input(
        self,
        input_shape: Optional[Tuple[int, ...]],
        input_sample: Optional[Callable[[], Any]],
    ) -> Any:
        """Prepare input for inference timing."""
        if input_sample is not None:
            inp = input_sample()
            return self._move_to_device(inp)

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

    def _move_to_device(self, inp: Any) -> Any:
        """Move input to the appropriate device."""
        if isinstance(inp, dict):
            return {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in inp.items()
            }
        elif isinstance(inp, (tuple, list)):
            return tuple((v.to(self.device) if torch.is_tensor(v) else v) for v in inp)
        elif torch.is_tensor(inp):
            return inp.to(self.device)
        return inp

    def _run_warmup(self, sample: Any, warmup_runs: int) -> None:
        """Run warmup iterations."""
        with torch.no_grad():
            for _ in range(warmup_runs):
                self._forward_pass(sample)

    def _measure_timing(self, sample: Any, num_runs: int) -> np.ndarray:
        """Measure inference timing."""
        times = []

        with torch.no_grad():
            if self.device.type == "cuda":
                times = self._measure_cuda_timing(sample, num_runs)
            else:
                times = self._measure_cpu_timing(sample, num_runs)

        return np.array(times)

    def _measure_cuda_timing(self, sample: Any, num_runs: int) -> list:
        """Measure timing using CUDA events."""
        times = []
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            self._forward_pass(sample)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            times.append(elapsed_ms / 1000.0)  # Convert to seconds

        return times

    def _measure_cpu_timing(self, sample: Any, num_runs: int) -> list:
        """Measure timing using time.time()."""
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            self._forward_pass(sample)
            end_time = time.time()
            times.append(end_time - start_time)

        return times

    def _forward_pass(self, sample: Any) -> Any:
        """Execute a forward pass with the given input."""
        if isinstance(sample, dict):
            return self.model(**sample)
        elif isinstance(sample, (tuple, list)):
            return self.model(*sample)
        else:
            return self.model(sample)

    def _compute_timing_statistics(self, times: np.ndarray) -> InferenceTimeInfo:
        """Compute timing statistics from measured times."""
        return InferenceTimeInfo(
            mean_time_ms=round(float(np.mean(times) * 1000), 4),
            std_time_ms=round(float(np.std(times) * 1000), 4),
            min_time_ms=round(float(np.min(times) * 1000), 4),
            max_time_ms=round(float(np.max(times) * 1000), 4),
            median_time_ms=round(float(np.median(times) * 1000), 4),
            runs=int(len(times)),
        )

    def analyze_attention_layers(self) -> AttentionLayerAnalysisInfo:
        """Analyze attention layers in the model."""
        attention_candidates = self._find_attention_candidates()
        leaf_candidates = self._filter_leaf_candidates(attention_candidates)
        attention_layers = self._extract_attention_info(
            leaf_candidates, attention_candidates
        )

        return AttentionLayerAnalysisInfo(
            num_attention_layers=len(attention_layers),
            attention_layers=attention_layers,
        )

    def _find_attention_candidates(self) -> Dict[str, nn.Module]:
        """Find modules that look like attention layers."""
        attention_candidates = {}

        for name, module in self.model.named_modules():
            if self._looks_like_attention(module):
                candidate_name = name if name else type(module).__name__
                attention_candidates[candidate_name] = module

        return attention_candidates

    def _looks_like_attention(self, module: nn.Module) -> bool:
        """Check if a module looks like an attention layer."""
        # Check if it's a known attention type
        if isinstance(module, torch.nn.MultiheadAttention):
            return True

        # Check for attention-related attributes
        attention_attrs = [
            "num_attention_heads",
            "num_heads",
            "all_head_size",
            "head_dim",
        ]
        if any(hasattr(module, attr) for attr in attention_attrs):
            return True

        # Check class name
        class_name = type(module).__name__.lower()
        attention_keywords = [
            "attention",
            "attn",
            "selfattn",
            "self_attention",
            "multihead",
        ]
        return any(keyword in class_name for keyword in attention_keywords)

    def _filter_leaf_candidates(self, candidates: Dict[str, nn.Module]) -> list:
        """Filter to get only leaf attention candidates."""
        candidate_names = list(candidates.keys())
        leaf_candidates = []

        for name in candidate_names:
            # Check if this candidate is a parent of any other candidate
            is_parent = any(
                (other != name) and other.startswith(name + ".")
                for other in candidate_names
            )
            if not is_parent:
                leaf_candidates.append(name)

        return sorted(leaf_candidates)

    def _extract_attention_info(
        self, leaf_candidates: list, candidates: Dict[str, nn.Module]
    ) -> list:
        """Extract information from attention layers."""
        attention_layers = []

        for name in leaf_candidates:
            module = candidates[name]
            info = {"name": name, "type": type(module).__name__}

            # Extract attention-specific attributes
            attention_attrs = [
                "num_attention_heads",
                "num_heads",
                "embed_dim",
                "head_dim",
            ]
            for attr in attention_attrs:
                if hasattr(module, attr):
                    info[attr] = getattr(module, attr)

            attention_layers.append(info)

        return attention_layers

    def get_model_summary(self) -> ModelSummary:
        """Get basic model summary information."""
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
        estimate_memory: bool = True,
        sequence_length: int = 2048,
        batch_size: int = 1,
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> LLMInfo:
        """
        Perform complete model profiling.

        Args:
            input_shape: Input shape for inference timing
            input_sample: Custom input sample function
            measure_inference: Whether to measure inference time
            estimate_memory: Whether to estimate memory requirements
            sequence_length: Sequence length for memory estimation
            batch_size: Batch size for memory estimation
            num_runs: Number of timing runs
            warmup_runs: Number of warmup runs

        Returns:
            Complete LLM profiling information
        """
        if self.verbose:
            logger.info(f"🔍 Profiling {self.model_name}...")

        # Basic profiling
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

        # Optional inference timing
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

        # Optional memory estimation
        memory_info = None
        if estimate_memory:
            memory_info = self.estimate_memory_requirements(
                sequence_length=sequence_length,
                batch_size=batch_size,
            )

        llm_info = LLMInfo(
            architecture=architecture_info,
            attention_layers=attention_layer_info,
            inference_time=inference_time_info,
            parameters=parameters_info,
            summary=model_summary,
            memory_estimation=memory_info,  # Add this to your LLMInfo dataclass
        )

        self.profile_data = llm_info
        if self.verbose:
            logger.info("✅ Profiling complete!")

        return llm_info

    def print_summary(self):
        print(self.profile_data.model_dump_json(indent=4))

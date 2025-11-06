from core.compression.engine.techniques.base.base_compression_technique import (
    BaseCompressionTechnique,
)
from core.compression.engine.techniques.quantization.transformers_lib_interface import (
    ITransformersQuantization,
)
from core.compression.engine.calibration.interface import ICalibration
from typing import List, Optional, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.analysis.profiling.profiler import Profiler
from core.analysis.profiling.models.profiler.llm_profile_options import (
    LLMProfilerOptions,
)
import gc
from loguru import logger
from pathlib import Path


class CompressionWorkflow:
    def __init__(
        self,
        hf_model_id: str,
        checkpoint_dir: Path,
        compression_technique: List[BaseCompressionTechnique],
        calibration: Optional[ICalibration] = None,
    ) -> None:
        self.hf_model_id = hf_model_id
        self.compression_technique = compression_technique
        self.calibration = calibration
        self.compressed_model: Optional[Any] = None
        self.compressed_tokenizer: Optional[Any] = None
        self.checkpoint_dir = checkpoint_dir

    def profile_base_model(
        self, profile_options: LLMProfilerOptions, verbose: bool = True
    ):
        model = AutoModelForCausalLM.from_pretrained(self.hf_model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        profiler = Profiler(model, tokenizer, self.hf_model_id, verbose=verbose)
        base_model_profile = profiler.profile_complete(
            model_profiler_options=profile_options
        )

        del model, tokenizer, profiler
        gc.collect()
        return base_model_profile

    def compress_model(self):
        count_quantization = 0
        quantization_idx: int | None = None

        for idx, technique in enumerate(self.compression_technique):
            if technique.technique_type == "quantization":
                count_quantization += 1
                quantization_idx = idx
                if count_quantization > 1:
                    raise ValueError(
                        "Only one quantization technique can be applied per compression workflow."
                    )
        model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_id, device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)

        float_techniques = [
            (idx, t)
            for idx, t in enumerate(self.compression_technique)
            if t.technique_type != "quantization"
        ]

        for idx, technique in float_techniques:
            logger.info(f"Applying {technique.technique_type}...")
            model = technique.compress(model=model, tokenizer=tokenizer)

        if quantization_idx is not None and float_techniques:
            logger.info("Saving pruned model before quantization...")
            step_name = "pre_quantization"
            self._save_checkpoint(model, tokenizer, step_name=step_name)
            logger.info("Loading pruned model with quantization config...")
            model, tokenizer = self._load_and_quantize(step_name=step_name)

        elif quantization_idx is not None:
            logger.info("Applying quantization...")
            quantization_technique: ITransformersQuantization = (
                self.compression_technique[quantization_idx]
            )
            model, tokenizer = quantization_technique.quantize(self.hf_model_id)
            self._save_checkpoint(model, tokenizer, step_name="final")

        self.compressed_model = model
        self.compressed_tokenizer = tokenizer

    def profile_compressed_model(
        self, profile_options: LLMProfilerOptions, verbose: bool = True
    ):
        if self.compressed_model is None or self.compressed_tokenizer is None:
            raise ValueError(
                "Model must be compressed before profiling the compressed model."
            )

        profiler = Profiler(
            self.compressed_model,
            self.compressed_tokenizer,
            model_name=f"{self.hf_model_id}-compressed",
            verbose=verbose,
        )
        compressed_model_profile = profiler.profile_complete(
            model_profiler_options=profile_options
        )

        return compressed_model_profile

    def _save_checkpoint(self, model: Any, tokenizer: Any, step_name: str):
        checkpoint_dir = Path(self.checkpoint_dir) / self.hf_model_id / step_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(checkpoint_dir / "model"))
        tokenizer.save_pretrained(str(checkpoint_dir / "tokenizer"))
        logger.info(f"✓ Checkpoint saved: {step_name}")

    def _load_checkpoint(self, step_name: str) -> Tuple[Any, Any]:
        checkpoint_dir = Path(self.checkpoint_dir) / self.hf_model_id / step_name

        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir / "model"), device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir / "tokenizer"))
        print(f"✓ Checkpoint loaded: {step_name}")
        return model, tokenizer

    def _load_and_quantize(self, step_name: str) -> Tuple[Any, Any]:
        """Load pruned model and apply quantization config."""
        checkpoint_dir = Path(self.checkpoint_dir) / self.hf_model_id / step_name

        quantization_idx = next(
            idx
            for idx, t in enumerate(self.compression_technique)
            if t.technique_type == "quantization"
        )
        quantization_technique: ITransformersQuantization = self.compression_technique[
            quantization_idx
        ]

        # Load base model path, then apply quantization
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir / "model"),
            quantization_config=quantization_technique.get_quantization_config(),
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir / "tokenizer"))

        self._save_checkpoint(model, tokenizer, step_name="quantized_model")
        return model, tokenizer

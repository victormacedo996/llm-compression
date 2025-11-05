from core.compression.engine.techniques.base.base_compression_technique import (
    BaseCompressionTechnique,
)
from core.compression.engine.techniques.quantization.transformers_lib_interface import (
    ITransformersQuantization,
)
from core.compression.engine.calibration.interface import ICalibration
from typing import List, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.analysis.profiling.profiler import Profiler
from core.analysis.profiling.models.profiler.llm_profile_options import (
    LLMProfilerOptions,
)
import gc


class CompressionWorkflow:
    def __init__(
        self,
        hf_model_id: str,
        compression_technique: List[BaseCompressionTechnique],
        calibration: Optional[ICalibration] = None,
    ) -> None:
        self.hf_model_id = hf_model_id
        self.compression_technique = compression_technique
        self.calibration = calibration
        self.compressed_model: Optional[Any] = None
        self.compressed_tokenizer: Optional[Any] = None

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

        if quantization_idx is not None:
            quantization_technique: ITransformersQuantization = (
                self.compression_technique[quantization_idx]
            )
            model, tokenizer = quantization_technique.quantize(self.hf_model_id)

        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_id, device_map="auto", trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)

        for technique in self.compression_technique:
            if technique.technique_type != "quantization":
                model = technique.compress(model=model, tokenizer=tokenizer)

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

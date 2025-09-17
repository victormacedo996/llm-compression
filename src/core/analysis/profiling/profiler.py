import torch.nn as nn
from core.analysis.profiling.hardware_profiler import HardwareProfiler
from core.analysis.profiling.llm_profiler import LLMProfiler
from core.analysis.profiling.models.profiler.llm_profile_options import (
    LLMProfilerOptions,
)
from core.analysis.profiling.models.profiler.complete_profile_output import (
    CompleteProfileOutput,
)


class Profiler:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        model_name: str = "my model",
        verbose: bool = True,
    ):
        self.hardware_profiler = HardwareProfiler()
        self.model_profiler = LLMProfiler(model, tokenizer, model_name, verbose)

    def profile_complete(
        self, model_profiler_options: LLMProfilerOptions | None = None
    ) -> CompleteProfileOutput:
        hardware_info = self.hardware_profiler.retrive_hardware_information()
        model_info = None

        if model_profiler_options:
            model_info = self.model_profiler.profile_complete(
                analyze_connections=model_profiler_options.analyze_connections,
                measure_inference=model_profiler_options.measure_inference_time,
                estimate_memory=model_profiler_options.estimate_memory,
            )

        return CompleteProfileOutput(
            hardware_profile=hardware_info, llm_profile=model_info
        )

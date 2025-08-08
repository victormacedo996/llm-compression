from pydantic import BaseModel
from core.analysis.profiling.models.llm import (
    ArchitectureInfo,
    InferenceTimeInfo,
    MemoryUsageInfo,
    ParameterInfo,
    ModelSummary,
    AttentionLayerAnalysisInfo,
)


class LLMInfo(BaseModel):
    parameters: ParameterInfo
    architecture: ArchitectureInfo
    memory_usage: MemoryUsageInfo | None = None
    inference_time: InferenceTimeInfo | None = None
    summary: ModelSummary
    attention_layers: AttentionLayerAnalysisInfo

from transformers import BertConfig, BertModel
from core.analysis.profiling.llm_profiler import LLMProfiler
import json
from core.analysis.profiling.models.llm import AnalyzeConnections, EstimateMemory

configuration = BertConfig()


model = BertModel(configuration)
# Example usage with your enhanced profiler
profiler = LLMProfiler(model, "My BERT Model", verbose=True)

# Profile with connection analysis
llm_info = profiler.profile_complete(
    analyze_connections=AnalyzeConnections(
        input_shape=(1, 512),
    ),
    estimate_memory=EstimateMemory(batch_size=1, sequence_length=512),
)

with open("test2.json", "w") as file:
    json.dump(llm_info.model_dump(), file, indent=4)

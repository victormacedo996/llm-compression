from pydantic import BaseModel
from typing import Tuple, Optional, Callable, Any


class MeasureInferenceTime(BaseModel):
    input_shape: Optional[Tuple[int, ...]]
    input_sample: Optional[Callable[[], Any]]
    num_runs: int
    warmup_runs: int

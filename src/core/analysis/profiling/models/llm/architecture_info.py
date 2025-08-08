from pydantic import BaseModel
from typing import Dict, Any, List


class ArchitectureInfo(BaseModel):
    total_layers: int
    layer_types_count: Dict[str, Any]
    layer_details: List[Dict[str, Any]]
    max_depth: Any | int

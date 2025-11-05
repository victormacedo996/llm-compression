from abc import ABC, abstractmethod
from typing import Any
from core.compression.engine.techniques.base.base_compression_technique import (
    BaseCompressionTechnique,
)


class IPrune(BaseCompressionTechnique, ABC):
    """Interface for quantization techniques used in compression engines."""

    technique_type: str = "prune"

    @abstractmethod
    def __init__(self, model: Any, technique_name: str, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def prune(self, model: Any, tokenizer: Any) -> Any:
        """Quantizes the LLM.

        Args:
            data: The data to be quantized.

        Returns:
            The quantized representation of the LLM.
        """
        raise NotImplementedError()

    def compress(self, *args, **kwargs):
        return self.prune(*args, **kwargs)

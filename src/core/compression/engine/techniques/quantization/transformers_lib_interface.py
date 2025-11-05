from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict
from core.compression.engine.techniques.base.base_compression_technique import (
    BaseCompressionTechnique,
)


class ITransformersQuantization(BaseCompressionTechnique, ABC):
    """Interface for quantization techniques used in compression engines."""

    technique_type: str = "quantization"

    @abstractmethod
    def __init__(
        self, technique_name: str, quantization_args: Dict[str, Any], **kwargs
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def quantize(self, hf_model_id: str, *args, **kwargs) -> Tuple[Any, Any]:
        """Quantizes the LLM.

        Args:
            hf_model_id: The Hugging Face model identifier.

        Returns:
            The quantized LLM with its tokenizer.
        """
        raise NotImplementedError()

    def compress(self, *args, **kwargs):
        return self.quantize(*args, **kwargs)

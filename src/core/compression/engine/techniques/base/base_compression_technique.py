from abc import ABC, abstractmethod
from typing import Any


class BaseCompressionTechnique(ABC):
    """Base class for compression techniques used in compression engines."""

    @property
    def technique_type() -> str:
        pass

    @abstractmethod
    def compress(self, *args, **kwargs) -> Any:
        """Compresses the model using the specified technique."""
        raise NotImplementedError()

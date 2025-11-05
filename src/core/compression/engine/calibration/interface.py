from abc import ABC, abstractmethod
from typing import Any, Tuple


class ICalibration(ABC):
    """Interface for calibration techniques used in compression engines."""

    @abstractmethod
    def calibrate(self) -> Tuple[Any, Any]:
        """Calibrates the model using the provided data.

        Args:
            data: The data to be used for calibration.

        Returns:
            The calibrated model with its tokenizer.
        """
        raise NotImplementedError()

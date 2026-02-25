from abc import ABC, abstractmethod
from ..eeg_data import EEGData


class BaseEEGExporter(ABC):
    @abstractmethod
    def export(self, data: EEGData, *args, **kwargs) -> None:
        """Write EEGData to a file or directory."""
        ...

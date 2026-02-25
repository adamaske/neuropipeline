from abc import ABC, abstractmethod
from ..eeg_data import EEGData


class BaseEEGImporter(ABC):
    @abstractmethod
    def load(self, filepath: str) -> EEGData:
        """Read a file and return a normalized EEGData object."""
        ...

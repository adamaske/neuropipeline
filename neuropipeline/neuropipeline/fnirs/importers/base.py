from abc import ABC, abstractmethod
from ..snirf_types import SNIRF


class BasefNIRSImporter(ABC):
    @abstractmethod
    def load(self, filepath: str) -> SNIRF:
        """Read a SNIRF file and return a normalized SNIRF object."""
        ...

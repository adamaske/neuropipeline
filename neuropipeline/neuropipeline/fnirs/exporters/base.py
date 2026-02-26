from abc import ABC, abstractmethod
from ..snirf_types import SNIRF


class BasefNIRSExporter(ABC):
    @abstractmethod
    def export(self, data: SNIRF, *args, **kwargs) -> None:
        """Write a SNIRF object to a file or directory."""
        ...

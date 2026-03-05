from .snirf_types import SNIRF
from .importers import SNIRFImporter
from .importers.base import BasefNIRSImporter
from .exporters import SNIRFExporter, fNIRSCSVExporter

_IMPORTERS: dict[str, type[BasefNIRSImporter]] = {
    "snirf": SNIRFImporter,
}


class fNIRSImporter:
    """
    Factory for loading fNIRS data from various source formats.

    Usage:
        data = fNIRSImporter.load("recording.snirf")
        data = fNIRSImporter.load("recording.snirf", source="snirf")
    """

    @staticmethod
    def load(filepath: str, source: str = "snirf") -> SNIRF:
        cls = _IMPORTERS.get(source)
        if cls is None:
            raise ValueError(
                f"Unknown fNIRS source '{source}'. Available: {list(_IMPORTERS)}"
            )
        return cls().load(filepath)


class fNIRSExporter:
    """
    Static export methods for writing SNIRF data to various formats.

    Usage:
        fNIRSExporter.to_snirf(data, output_path="out.snirf", source_filepath="original.snirf")
        fNIRSExporter.to_csv(data, output_folder="out/", name="subject_01")
    """

    @staticmethod
    def to_snirf(data: SNIRF, output_path: str) -> None:
        """
        Export to SNIRF format, written from scratch.

        Args:
            data:        SNIRF object to export.
            output_path: Destination .snirf file path.
        """
        SNIRFExporter().export(data, output_path)

    @staticmethod
    def to_csv(data: SNIRF, output_folder: str, name: str | None = None) -> None:
        """
        Export to CSV files.

        Writes {name}_channels.csv (time + channel columns) and
        {name}_events.csv (onset_s, event_name, duration_s, value) into output_folder.

        Args:
            data:          SNIRF object to export.
            output_folder: Directory to write files into.
            name:          Base filename stem. Defaults to "fnirs".
        """
        fNIRSCSVExporter().export(data, output_folder, name)

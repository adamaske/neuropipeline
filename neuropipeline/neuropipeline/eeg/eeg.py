from .eeg_data import EEGData
from .importers import GRecorderImporter
from .importers.base import BaseEEGImporter
from .exporters import EEGLABExporter, HDF5Exporter

_IMPORTERS: dict[str, type[BaseEEGImporter]] = {
    "gRecorder": GRecorderImporter,
}


class EEGImporter:
    """
    Factory for loading EEG data from various source formats.

    Usage:
        eeg = EEGImporter.load("recording.hdf5", source="gRecorder")
    """

    @staticmethod
    def load(filepath: str, source: str) -> EEGData:
        cls = _IMPORTERS.get(source)
        if cls is None:
            raise ValueError(
                f"Unknown EEG source '{source}'. Available: {list(_IMPORTERS)}"
            )
        return cls().load(filepath)


class EEGExporter:
    """
    Static export methods for writing EEGData to various formats.

    Usage:
        EEGExporter.to_eeglab(eeg, output_folder="out/", name="subject_01")
        EEGExporter.to_hdf5(eeg, output_path="out.hdf5", source_filepath="original.hdf5")
    """

    @staticmethod
    def to_eeglab(data: EEGData, output_folder: str, name: str | None = None) -> None:
        """
        Export to EEGLAB .set + .fdt format.

        Args:
            data: EEGData to export.
            output_folder: Directory to write files into.
            name: Base filename (without extension). Defaults to "eeg".
        """
        EEGLABExporter().export(data, output_folder, name)

    @staticmethod
    def to_hdf5(data: EEGData, output_path: str, source_filepath: str) -> None:
        """
        Export back to g.Recorder HDF5 format.

        Copies the source file structure and overwrites channel data and features.

        Args:
            data: EEGData to export.
            output_path: Destination .hdf5 file path.
            source_filepath: Original .hdf5 file to copy structure from.
        """
        HDF5Exporter().export(data, output_path, source_filepath)

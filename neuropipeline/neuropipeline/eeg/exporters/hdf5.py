import os
import shutil
import numpy as np
import h5py

from .base import BaseEEGExporter
from ..eeg_data import EEGData


class HDF5Exporter(BaseEEGExporter):
    """
    Exports EEGData back to a g.Recorder-compatible HDF5 file.

    Requires the original source .hdf5 file to copy structure from,
    since the gRecorder format embeds XML metadata that we do not regenerate.
    Feature onsets are converted back from seconds → samples for storage.
    """

    def export(self, data: EEGData, output_path: str, source_filepath: str) -> None:
        stem, suffix = os.path.splitext(output_path)
        temp_path = stem + "_temporary" + suffix

        shutil.copy(source_filepath, temp_path)
        print(f"HDF5Exporter: copied {source_filepath} → {temp_path}")

        with h5py.File(temp_path, mode="r+") as hdf:
            # Overwrite channel data (transpose back to samples × channels)
            rawdata = hdf["RawData"]
            del rawdata["Samples"]
            rawdata.create_dataset("Samples", data=data.channel_data.T, dtype="f4")

            # Overwrite features (convert seconds → samples)
            onsets_samples = (data.feature_onsets * data.sampling_frequency).astype(np.int64)
            async_group = hdf["AsynchronData"]

            if "Time" in async_group:
                del async_group["Time"]
            if "TypeID" in async_group:
                del async_group["TypeID"]

            async_group.create_dataset(
                "Time",
                data=onsets_samples.reshape(-1, 1),
            )
            async_group.create_dataset(
                "TypeID",
                data=np.array(data.feature_descriptions).reshape(-1, 1),
            )

        if os.path.exists(output_path):
            ans = input(f"{output_path} already exists. Overwrite? [Y/N]: ")
            if ans.strip().upper() == "Y":
                shutil.move(temp_path, output_path)
                print(f"HDF5Exporter: wrote {output_path}")
            else:
                os.remove(temp_path)
                print("HDF5Exporter: export cancelled.")
        else:
            os.rename(temp_path, output_path)
            print(f"HDF5Exporter: wrote {output_path}")

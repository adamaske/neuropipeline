import os

import numpy as np

from .base import BaseEEGExporter
from ..eeg_data import EEGData


class EEGCSVExporter(BaseEEGExporter):
    """
    Exports EEGData to CSV files.

    Writes two files into output_folder:
        {name}_channels.csv  — time vector + one column per channel
        {name}_events.csv    — onset_s, description
    """

    def export(self, data: EEGData, output_folder: str, name: str | None = None) -> None:
        os.makedirs(output_folder, exist_ok=True)
        stem = name or "eeg"

        # --- Channel data ---
        channels_path = os.path.join(output_folder, stem + "_channels.csv")
        time = data.get_time()
        header = "time," + ",".join(data.channel_names)
        channel_matrix = np.column_stack([time, data.channel_data.T])
        np.savetxt(channels_path, channel_matrix, delimiter=",", header=header, comments="")
        print(f"EEGCSVExporter: wrote {channels_path}")

        # --- Events ---
        events_path = os.path.join(output_folder, stem + "_events.csv")
        with open(events_path, "w") as f:
            f.write("onset_s,description\n")
            for onset, desc in zip(data.feature_onsets, data.feature_descriptions):
                f.write(f"{onset},{desc}\n")
        print(f"EEGCSVExporter: wrote {events_path}")

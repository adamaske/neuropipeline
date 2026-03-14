"""
CSV exporter for fNIRS data.

Writes two files:
    {name}_channels.csv — time vector + one column per channel
    {name}_events.csv   — onset_s, event_name, duration_s
"""

import os

import numpy as np


class fNIRSCSVExporter:

    def export(self, fnirs, output_folder: str, name: str | None = None) -> None:
        os.makedirs(output_folder, exist_ok=True)
        stem = name or "fnirs"

        raw = fnirs.raw

        # ---- Channel data ----
        channels_path = os.path.join(output_folder, stem + "_channels.csv")
        data = raw.get_data().T  # (samples, channels)
        times = raw.times
        header = "time," + ",".join(raw.ch_names)
        matrix = np.column_stack([times, data])
        np.savetxt(channels_path, matrix, delimiter=",", header=header, comments="")
        print(f"fNIRSCSVExporter: wrote {channels_path}")

        # ---- Events ----
        events_path = os.path.join(output_folder, stem + "_events.csv")
        with open(events_path, "w") as f:
            f.write("onset_s,event_name,duration_s\n")
            for ann in raw.annotations:
                f.write(f"{ann['onset']},{ann['description']},{ann['duration']}\n")
        print(f"fNIRSCSVExporter: wrote {events_path}")

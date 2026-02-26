import os

import numpy as np

from .base import BasefNIRSExporter
from ..snirf_types import SNIRF


class fNIRSCSVExporter(BasefNIRSExporter):
    """
    Exports a SNIRF object to CSV files.

    Writes two files into output_folder:
        {name}_channels.csv  — time vector + one column per channel
        {name}_events.csv    — onset_s, event_name, duration_s, value
    """

    def export(self, data: SNIRF, output_folder: str, name: str | None = None) -> None:
        os.makedirs(output_folder, exist_ok=True)
        stem = name or "fnirs"

        channel_names = data.get_channel_names()
        time          = data.get_time()

        # ---- Channel data ----
        channels_path = os.path.join(output_folder, stem + "_channels.csv")
        header = "time," + ",".join(channel_names)
        matrix = np.column_stack([time, data.channel_store.data.T])
        np.savetxt(channels_path, matrix, delimiter=",", header=header, comments="")
        print(f"fNIRSCSVExporter: wrote {channels_path}")

        # ---- Events ----
        events_path = os.path.join(output_folder, stem + "_events.csv")
        with open(events_path, "w") as f:
            f.write("onset_s,event_name,duration_s,value\n")
            for event in data.events.events:
                for marker in event.markers:
                    f.write(f"{marker.onset},{event.name},{marker.duration},{marker.value}\n")
        print(f"fNIRSCSVExporter: wrote {events_path}")

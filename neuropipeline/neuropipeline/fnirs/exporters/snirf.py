import os
import shutil

import h5py
import numpy as np

from .base import BasefNIRSExporter
from ..snirf_types import SNIRF


class SNIRFExporter(BasefNIRSExporter):
    """
    Exports a SNIRF object to a .snirf (HDF5) file.

    Requires the original source .snirf file to copy structure from — this
    preserves probe geometry, metadata, and format-specific datasets that are
    not reconstructed from the SNIRF struct.

    Overwrites:
        /nirs/data1/dataTimeSeries  — channel_store.data (transposed back to
                                      [timepoints × channels] for HDF5 layout)
        /nirs/data1/time            — time_data.time
        /nirs/stim{i}               — rebuilt from events
    """

    def export(self, data: SNIRF, output_path: str, source_filepath: str) -> None:
        stem, suffix = os.path.splitext(output_path)
        temp_path = stem + "_temporary" + suffix

        shutil.copy(source_filepath, temp_path)
        print(f"SNIRFExporter: copied {source_filepath} → {temp_path}")

        with h5py.File(temp_path, "r+") as f:
            nirs_keys = [k for k in f.keys() if k.startswith("nirs")]
            nirs_key = "nirs" if "nirs" in nirs_keys else sorted(nirs_keys)[0]
            nirs = f[nirs_key]

            # ---- Overwrite dataTimeSeries and time ----
            if "data1" in nirs:
                dg = nirs["data1"]

                # HDF5 expects [timepoints × channels]
                ts = data.channel_store.data.T  # (timepoints, channels)

                if "dataTimeSeries" in dg:
                    del dg["dataTimeSeries"]
                dg.create_dataset("dataTimeSeries", data=ts.astype(np.float64))

                if "time" in dg:
                    del dg["time"]
                dg.create_dataset("time", data=data.time_data.time.astype(np.float64))

            # ---- Overwrite stim blocks ----
            for key in [k for k in nirs.keys() if k.startswith("stim")]:
                del nirs[key]

            for i, event in enumerate(data.events.events, start=1):
                if not event.markers:
                    continue
                sg = nirs.create_group(f"stim{i}")
                stim_data = np.array(
                    [[m.onset, m.duration, m.value] for m in event.markers],
                    dtype=np.float64,
                )
                sg.create_dataset("data", data=stim_data)
                sg.create_dataset("name", data=event.name.encode("utf-8"))

        if os.path.exists(output_path):
            ans = input(f"{output_path} already exists. Overwrite? [Y/N]: ")
            if ans.strip().upper() == "Y":
                shutil.move(temp_path, output_path)
                print(f"SNIRFExporter: wrote {output_path}")
            else:
                os.remove(temp_path)
                print("SNIRFExporter: export cancelled.")
        else:
            os.rename(temp_path, output_path)
            print(f"SNIRFExporter: wrote {output_path}")

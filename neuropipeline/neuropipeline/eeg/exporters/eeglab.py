import os
import numpy as np
import scipy.io as sio

from .base import BaseEEGExporter
from ..eeg_data import EEGData


class EEGLABExporter(BaseEEGExporter):
    """
    Exports EEGData to EEGLAB-compatible .set + .fdt format.

    .fdt  — raw float32 binary, channels × samples (C-order)
    .set  — MATLAB struct written via scipy.io.savemat
    """

    def export(self, data: EEGData, output_folder: str, name: str | None = None) -> None:
        os.makedirs(output_folder, exist_ok=True)

        stem = name or "eeg"
        set_path = os.path.join(output_folder, stem + ".set")
        fdt_path = os.path.join(output_folder, stem + ".fdt")
        fdt_filename = stem + ".fdt"

        n_samples = data.channel_data.shape[1]
        duration = data.get_duration()

        # --- Write .fdt (raw float32, channels × samples) ---
        data.channel_data.astype(np.float32).tofile(fdt_path)
        print(f"EEGLABExporter: wrote {fdt_path}")

        # --- Build chanlocs struct array ---
        chanlocs_dtype = np.dtype([("labels", "O"), ("type", "O")])
        chanlocs = np.zeros(data.channel_num, dtype=chanlocs_dtype)
        for i, name_ in enumerate(data.channel_names):
            chanlocs[i]["labels"] = name_
            chanlocs[i]["type"] = "EEG"

        # --- Build event / urevent struct arrays ---
        n_events = len(data.feature_onsets)
        event_dtype = np.dtype([
            ("type", "O"),
            ("latency", "f8"),    # 1-indexed sample number (EEGLAB convention)
            ("duration", "f8"),
            ("urevent", "f8"),
        ])

        if n_events > 0:
            events = np.zeros(n_events, dtype=event_dtype)
            for i, (onset_sec, desc) in enumerate(
                zip(data.feature_onsets, data.feature_descriptions)
            ):
                events[i]["type"] = str(int(desc))
                events[i]["latency"] = onset_sec * data.sampling_frequency + 1  # 1-indexed
                events[i]["duration"] = 0.0
                events[i]["urevent"] = float(i + 1)
        else:
            events = np.zeros(0, dtype=event_dtype)

        # --- Build EEG struct dict ---
        eeg_struct = {
            "setname":    stem,
            "filename":   stem + ".set",
            "filepath":   output_folder,
            "data":       fdt_filename,
            "nbchan":     float(data.channel_num),
            "pnts":       float(n_samples),
            "trials":     1.0,
            "srate":      float(data.sampling_frequency),
            "xmin":       0.0,
            "xmax":       float(duration),
            "times":      (np.arange(n_samples) / data.sampling_frequency * 1000.0),  # ms
            "chanlocs":   chanlocs,
            "event":      events,
            "urevent":    events.copy(),
            "ref":        "common",
            "comments":   data.session_comment or "",
            "icawinv":    np.array([]),
            "icasphere":  np.array([]),
            "icaweights": np.array([]),
            "icachansind": np.array([]),
            "icaact":     np.array([]),
            "saved":      "yes",
        }

        sio.savemat(set_path, {"EEG": eeg_struct}, format="5", do_compression=False)
        print(f"EEGLABExporter: wrote {set_path}")

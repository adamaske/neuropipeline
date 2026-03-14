"""
SNIRF exporter for fNIRS data.

Wraps ``mne_nirs.io.write_raw_snirf`` to export any stage of processing
(raw CW amplitude, optical density, or haemoglobin) to SNIRF format v1.1.
"""

import mne_nirs.io


class fNIRSSNIRFExporter:

    def export(self, fnirs, filepath: str, add_montage: bool = False) -> None:
        mne_nirs.io.write_raw_snirf(fnirs.raw, filepath, add_montage=add_montage)
        print(f"fNIRSSNIRFExporter: wrote {filepath}")
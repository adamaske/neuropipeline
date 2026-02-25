import h5py
import numpy as np
import xml.etree.ElementTree as et

from .base import BaseEEGImporter
from ..eeg_data import EEGData


def _parse_xml(xml_str: str) -> dict:
    root = et.fromstring(xml_str)
    return {child.tag: child.text.strip() if child.text else None for child in root}


class GRecorderImporter(BaseEEGImporter):
    """Imports EEG data from g.Recorder HDF5 files."""

    def load(self, filepath: str) -> EEGData:
        print(f"GRecorderImporter: reading {filepath}")

        with h5py.File(filepath, mode="r") as hdf:
            rawdata = hdf["RawData"]

            # Channel data: stored (samples × channels), transpose to (channels × samples)
            channel_data = np.array(rawdata["Samples"]).transpose()

            # Features / markers
            async_data = hdf["AsynchronData"]
            if async_data.get("Time") is not None:
                onsets_samples = np.array(async_data["Time"]).T[0]
                feature_descriptions = np.array(async_data["TypeID"]).T[0]
            else:
                onsets_samples = np.array([], dtype=float)
                feature_descriptions = np.array([], dtype=int)

            # Acquisition task description
            acq_desc = _parse_xml(rawdata["AcquisitionTaskDescription"].asstr()[0])
            channel_num = int(acq_desc["NumberOfAcquiredChannels"])
            sampling_frequency = float(acq_desc["SamplingFrequency"])

            # DAQ device description
            daq_desc = _parse_xml(rawdata["DAQDeviceDescription"].asstr()[0])
            acquisition_unit = daq_desc.get("Unit")
            acquisition_device = daq_desc.get("Name")

            # Session description
            session_desc = _parse_xml(rawdata["SessionDescription"].asstr()[0])
            session_run = session_desc.get("Run")
            session_comment = session_desc.get("Comment")

        # Normalize onsets: samples → seconds
        feature_onsets = onsets_samples / sampling_frequency

        # gRecorder does not provide channel names — auto-generate
        channel_names = [f"CH{i + 1}" for i in range(channel_num)]

        return EEGData(
            channel_data=channel_data,
            channel_names=channel_names,
            channel_num=channel_num,
            sampling_frequency=sampling_frequency,
            feature_onsets=feature_onsets,
            feature_descriptions=feature_descriptions,
            session_run=session_run,
            session_comment=session_comment,
            acquisition_device=acquisition_device,
            acquisition_unit=acquisition_unit,
        )

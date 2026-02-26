from dataclasses import dataclass
from enum import Enum

import numpy as np


class fnirs_data_type(Enum):
    Wavelength = "Wavelength"
    OpticalDensity = "Optical Density"
    HemoglobinConcentration = "Hemoglobin Concentration"


WL = fnirs_data_type.Wavelength
OD = fnirs_data_type.OpticalDensity
CC = fnirs_data_type.HemoglobinConcentration


@dataclass
class fNIRSData:
    """
    Format-agnostic fNIRS data container.

    All temporal values are normalized to seconds regardless of source format.
    channel_data is always (channels × samples).
    """
    channel_data: np.ndarray          # (channels × samples)
    channel_names: list[str]
    channel_num: int
    sampling_frequency: float         # Hz
    feature_onsets: np.ndarray        # seconds
    feature_descriptions: np.ndarray
    feature_durations: np.ndarray     # seconds
    data_type: fnirs_data_type = WL

    def get_duration(self) -> float:
        """Total recording duration in seconds."""
        return self.channel_data.shape[1] / self.sampling_frequency

    def get_time(self) -> np.ndarray:
        """Time vector in seconds starting from 0."""
        return np.arange(self.channel_data.shape[1]) / self.sampling_frequency

    def print(self):
        print("fNIRSData")
        print(f"  data_type          : {self.data_type.value}")
        print(f"  sampling_frequency : {self.sampling_frequency} Hz")
        print(f"  channel_num        : {self.channel_num}")
        print(f"  channel_data       : {self.channel_data.shape}")
        print(f"  channel_names      : {self.channel_names}")
        print(f"  feature_onsets     : {self.feature_onsets}")
        print(f"  feature_descriptions: {self.feature_descriptions}")
        print(f"  feature_durations  : {self.feature_durations}")
        print(f"  duration           : {self.get_duration():.2f}s")

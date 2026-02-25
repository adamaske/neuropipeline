from dataclasses import dataclass, field
import numpy as np


@dataclass
class EEGData:
    """
    Format-agnostic EEG data container.

    All temporal values are normalized to seconds regardless of source format.
    channel_data is always (channels × samples).
    """
    channel_data: np.ndarray          # (channels × samples)
    channel_names: list[str]
    channel_num: int
    sampling_frequency: float         # Hz
    feature_onsets: np.ndarray        # seconds
    feature_descriptions: np.ndarray
    session_run: str | None = None
    session_comment: str | None = None
    acquisition_device: str | None = None
    acquisition_unit: str | None = None

    def get_duration(self) -> float:
        """Total recording duration in seconds."""
        return self.channel_data.shape[1] / self.sampling_frequency

    def get_time(self) -> np.ndarray:
        """Time vector in seconds starting from 0."""
        return np.arange(self.channel_data.shape[1]) / self.sampling_frequency

    def print(self):
        print("EEGData")
        print(f"  sampling_frequency : {self.sampling_frequency} Hz")
        print(f"  channel_num        : {self.channel_num}")
        print(f"  channel_data       : {self.channel_data.shape}")
        print(f"  channel_names      : {self.channel_names}")
        print(f"  feature_onsets     : {self.feature_onsets}")
        print(f"  feature_descriptions: {self.feature_descriptions}")
        print(f"  duration           : {self.get_duration():.2f}s")
        print(f"  session_run        : {self.session_run}")
        print(f"  session_comment    : {self.session_comment}")
        print(f"  acquisition_device : {self.acquisition_device}")
        print(f"  acquisition_unit   : {self.acquisition_unit}")

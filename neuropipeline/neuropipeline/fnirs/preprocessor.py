"""
Preprocessing pipeline configuration for fNIRS data.

Provides the ``fNIRSPreprocessor`` class — a declarative pipeline that
delegates all processing to MNE-Python and MNE-NIRS.
"""

from __future__ import annotations


class fNIRSPreprocessor:
    """
    Configurable preprocessing pipeline for fNIRS data.

    Steps are executed in the standard scientific order:

    1. Raw intensity -> Optical density
    2. TDDR motion correction
    3. Short-channel regression (systemic artifact removal)
    4. Optical density -> Haemoglobin concentration (Beer-Lambert)
    5. Bandpass temporal filtering

    Usage::

        pp = fNIRSPreprocessor()
        pp.set_bandpass(0.01, 0.2)
        pp.set_motion_correction(False)
        fnirs.preprocess(pp)

    Or fluent::

        fNIRSPreprocessor().set_bandpass(0.01, 0.2).apply(fnirs)
    """

    def __init__(
        self,
        optical_density: bool = True,
        motion_correction: bool = True,
        hemoglobin: bool = True,
        short_channel_regression: bool = False,
        bandpass: bool = True,
        bandpass_low: float = 0.01,
        bandpass_high: float = 0.1,
        ppf: float = 6.0,
    ):
        self.optical_density = optical_density
        self.motion_correction = motion_correction
        self.hemoglobin = hemoglobin
        self.short_channel_regression = short_channel_regression
        self.bandpass = bandpass
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.ppf = ppf

    # -- Builder methods (each returns self) --

    def set_optical_density(self, enabled: bool = True) -> "fNIRSPreprocessor":
        self.optical_density = enabled
        return self

    def set_motion_correction(self, enabled: bool = True) -> "fNIRSPreprocessor":
        self.motion_correction = enabled
        return self

    def set_hemoglobin(self, enabled: bool = True, ppf: float = 6.0) -> "fNIRSPreprocessor":
        self.hemoglobin = enabled
        self.ppf = ppf
        return self

    def set_short_channel_regression(self, enabled: bool = True) -> "fNIRSPreprocessor":
        self.short_channel_regression = enabled
        return self

    def set_bandpass(self, low: float = 0.01, high: float = 0.1, enabled: bool = True) -> "fNIRSPreprocessor":
        self.bandpass = enabled
        self.bandpass_low = low
        self.bandpass_high = high
        return self

    # -- Apply --

    def apply(self, fnirs) -> object:
        """Apply the configured pipeline to an ``fNIRS`` object.

        Returns the ``fNIRS`` object so callers can continue chaining.
        """
        if self.optical_density:
            fnirs.to_optical_density()
        if self.motion_correction:
            fnirs.tddr()
        if self.short_channel_regression:
            fnirs.short_channel_regression()
        if self.hemoglobin:
            fnirs.to_hemoglobin(ppf=self.ppf)
        if self.bandpass:
            fnirs.bandpass(self.bandpass_low, self.bandpass_high)
        return fnirs

    # -- Display --

    def print(self) -> None:
        """Print current pipeline settings."""
        print("fNIRSPreprocessor Settings:")
        print(f"  optical_density:          {self.optical_density}")
        print(f"  motion_correction (TDDR): {self.motion_correction}")
        print(f"  short_channel_regression: {self.short_channel_regression}")
        print(f"  hemoglobin (Beer-Lambert):{self.hemoglobin} (ppf={self.ppf})")
        print(f"  bandpass:                 {self.bandpass} "
              f"({self.bandpass_low}-{self.bandpass_high} Hz)")

    def __repr__(self) -> str:
        steps = []
        if self.optical_density:
            steps.append("OD")
        if self.motion_correction:
            steps.append("TDDR")
        if self.short_channel_regression:
            steps.append("SCR")
        if self.hemoglobin:
            steps.append("HbX")
        if self.bandpass:
            steps.append(f"BP({self.bandpass_low}-{self.bandpass_high})")
        return f"fNIRSPreprocessor([{' -> '.join(steps)}])"

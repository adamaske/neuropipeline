"""
fNIRS processing class — MNE-NIRS wrapper.

Provides a simplified, chainable API around MNE-Python and MNE-NIRS
for fNIRS data loading, preprocessing, and export.
"""

from __future__ import annotations

import mne
import mne.preprocessing.nirs
import numpy as np

mne.set_log_level("WARNING")


# ---------------------------------------------------------------------------
# FFT / PSD helpers (standalone, used by visualizer)
# ---------------------------------------------------------------------------

def compute_fft(time_series, fs, freq_limit: float | None):
    N = len(time_series)
    fft_result = np.fft.fft(time_series)
    fft_freq = np.fft.fftfreq(N, d=1 / fs)

    positive_freqs = fft_freq[:N // 2]
    positive_spectrum = np.abs(fft_result[:N // 2]) * (2 / N)

    if freq_limit is None:
        return positive_freqs, positive_spectrum

    indices = positive_freqs <= freq_limit
    return positive_freqs[indices], positive_spectrum[indices]


def compute_psd(time_series, fs, freq_limit: float | None):
    freqs, spectrum = compute_fft(time_series, fs, freq_limit)
    psd = np.square(spectrum) / (fs * len(time_series))
    psd[1:] = 2 * psd[1:]
    return freqs, psd



# ---------------------------------------------------------------------------
# fNIRS class
# ---------------------------------------------------------------------------

class fNIRS:
    """
    Simplified fNIRS processing class wrapping MNE-Python and MNE-NIRS.

    All preprocessing methods return ``self`` for chaining::

        f = (fNIRS("file.snirf")
             .to_optical_density()
             .tddr()
             .to_hemoglobin()
             .bandpass(0.01, 0.1))

    Access the underlying MNE Raw object via ``.raw`` for advanced operations.
    """

    def __init__(self, filepath: str | None = None):
        self._raw: mne.io.Raw | None = None
        self._filepath: str | None = None
        if filepath is not None:
            self.load(filepath)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def raw(self) -> mne.io.Raw:
        """The underlying MNE Raw object."""
        if self._raw is None:
            raise RuntimeError("No data loaded. Call load() first.")
        return self._raw

    @property
    def sampling_frequency(self) -> float:
        return self.raw.info["sfreq"]

    @property
    def channel_names(self) -> list[str]:
        return self.raw.ch_names

    @property
    def channel_count(self) -> int:
        return len(self.raw.ch_names)

    @property
    def duration(self) -> float:
        return self.raw.times[-1]

    @property
    def data_type(self) -> str:
        """Current data type based on MNE channel types.

        Returns one of ``'raw'``, ``'optical_density'``, or ``'hemoglobin'``.
        """
        ch_types = set(self.raw.get_channel_types())
        if ch_types & {"hbo", "hbr"}:
            return "hemoglobin"
        if "fnirs_od" in ch_types:
            return "optical_density"
        return "raw"

    @property
    def annotations(self) -> mne.Annotations:
        """MNE Annotations (events / stimuli)."""
        return self.raw.annotations

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load(self, filepath: str) -> "fNIRS":
        """Load fNIRS data from file.

        Supports SNIRF (``.snirf``) and MNE FIF (``.fif``) formats.
        """
        if filepath.endswith(".snirf"):
            self._raw = mne.io.read_raw_snirf(filepath, preload=True)
        elif filepath.endswith((".fif", ".fif.gz")):
            self._raw = mne.io.read_raw_fif(filepath, preload=True)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        self._filepath = filepath
        return self

    def save(self, filepath: str, overwrite: bool = False) -> "fNIRS":
        """Save data to file.

        Supports MNE FIF format (``.fif``) natively. For other formats
        (EDF, BrainVision), delegates to ``mne.export.export_raw``.
        """
        if filepath.endswith((".fif", ".fif.gz")):
            self.raw.save(filepath, overwrite=overwrite)
        else:
            mne.export.export_raw(filepath, self.raw, fmt="auto", overwrite=overwrite)
        return self

    def to_csv(self, output_folder: str, name: str | None = None) -> "fNIRS":
        """Export channel data and events to CSV files."""
        from .exporters.csv import fNIRSCSVExporter
        fNIRSCSVExporter().export(self, output_folder, name)
        return self

    def to_snirf(self, filepath: str, add_montage: bool = False) -> "fNIRS":
        """Export data to SNIRF format v1.1.

        Wraps ``mne_nirs.io.write_raw_snirf()``. Works at any processing
        stage: raw CW amplitude, optical density, or haemoglobin.

        Args:
            filepath: Output path, e.g. ``"output/processed.snirf"``.
            add_montage: If True, adds montage as landmarks for AtlasViewer
                compatibility (default False).
        """
        from .exporters.snirf import fNIRSSNIRFExporter
        fNIRSSNIRFExporter().export(self, filepath, add_montage=add_montage)
        return self

    # ------------------------------------------------------------------
    # Preprocessing (each returns self for chaining)
    # ------------------------------------------------------------------

    def to_optical_density(self) -> "fNIRS":
        """Convert raw intensity to optical density.

        Wraps ``mne.preprocessing.nirs.optical_density()``.
        """
        self._raw = mne.preprocessing.nirs.optical_density(self._raw)
        return self

    def to_hemoglobin(self, ppf: float = 6.0) -> "fNIRS":
        """Convert optical density to haemoglobin concentration (Beer-Lambert).

        Wraps ``mne.preprocessing.nirs.beer_lambert_law()``.

        Args:
            ppf: Partial pathlength factor (default 6.0 for adult brain).
        """
        self._raw = mne.preprocessing.nirs.beer_lambert_law(self._raw, ppf=ppf)
        return self

    def tddr(self) -> "fNIRS":
        """Apply TDDR motion correction.

        Wraps ``mne.preprocessing.nirs.temporal_derivative_distribution_repair()``.
        """
        self._raw = mne.preprocessing.nirs.temporal_derivative_distribution_repair(self._raw)
        return self

    def bandpass(self, low: float = 0.01, high: float = 0.1) -> "fNIRS":
        """Apply bandpass filter.

        Wraps ``raw.filter()``.
        """
        self._raw.filter(
        l_freq=low,
        h_freq=high,
        method='fir',
        phase='zero-double',
        h_trans_bandwidth=0.02,   # tight roll-off: stopband starts at 0.12 Hz
        l_trans_bandwidth=0.005,  # tight roll-off on lowpass side
        )
        return self

    def short_channel_regression(self) -> "fNIRS":
        """Remove systemic artifacts via short-channel regression.

        Wraps ``mne_nirs.signal_enhancement.short_channel_regression()``.
        """
        import mne_nirs.signal_enhancement
        self._raw = mne_nirs.signal_enhancement.short_channel_regression(self._raw)
        return self

    def resample(self, sfreq: float) -> "fNIRS":
        """Resample data to a new sampling frequency."""
        self._raw.resample(sfreq)
        return self

    def crop(self, tmin: float = 0.0, tmax: float | None = None) -> "fNIRS":
        """Crop the recording to ``[tmin, tmax]`` seconds."""
        self._raw.crop(tmin=tmin, tmax=tmax)
        return self

    def preprocess(self, preprocessor: object | None = None, **kwargs) -> "fNIRS":
        """Apply a full preprocessing pipeline.

        Pass an ``fNIRSPreprocessor`` instance or keyword arguments that will
        be forwarded to a new ``fNIRSPreprocessor``.
        """
        from .preprocessor import fNIRSPreprocessor
        if preprocessor is None:
            preprocessor = fNIRSPreprocessor(**kwargs)
        preprocessor.apply(self)
        return self

    # ------------------------------------------------------------------
    # Channel operations
    # ------------------------------------------------------------------

    def pick_long_channels(self, min_dist: float = 0.01) -> "fNIRS":
        """Keep only long-separation channels (distance >= *min_dist* m)."""
        import mne_nirs.channels
        picks = mne_nirs.channels.get_long_channels(self._raw, min_dist=min_dist)
        self._raw.pick(picks)
        return self

    def pick_short_channels(self, max_dist: float = 0.01) -> "fNIRS":
        """Keep only short-separation channels (distance < *max_dist* m)."""
        import mne_nirs.channels
        picks = mne_nirs.channels.get_short_channels(self._raw, max_dist=max_dist)
        self._raw.pick(picks)
        return self

    def get_data(self, picks: str | list[str] | None = None) -> np.ndarray:
        """Get channel data as a numpy array (channels x samples)."""
        return self.raw.get_data(picks=picks)

    def get_hbo(self) -> tuple[np.ndarray, list[str]]:
        """Get HbO channel data and names."""
        picks = mne.pick_types(self.raw.info, fnirs="hbo")
        names = [self.raw.ch_names[i] for i in picks]
        return self.raw.get_data(picks=picks), names

    def get_hbr(self) -> tuple[np.ndarray, list[str]]:
        """Get HbR channel data and names."""
        picks = mne.pick_types(self.raw.info, fnirs="hbr")
        names = [self.raw.ch_names[i] for i in picks]
        return self.raw.get_data(picks=picks), names

    # ------------------------------------------------------------------
    # Events / Epochs
    # ------------------------------------------------------------------

    def events(self, event_id: dict | None = None) -> tuple[np.ndarray, dict]:
        """Extract events array from annotations.

        Returns ``(events, event_id)`` tuple compatible with ``mne.Epochs``.
        """
        return mne.events_from_annotations(self.raw, event_id=event_id)

    def epochs(
        self,
        event_id: dict | None = None,
        tmin: float = -0.5,
        tmax: float = 5.0,
        baseline: tuple | None = (None, 0),
    ) -> mne.Epochs:
        """Create MNE Epochs from annotations."""
        events, eid = self.events(event_id)
        return mne.Epochs(
            self.raw, events, event_id=eid,
            tmin=tmin, tmax=tmax, baseline=baseline,
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(self, **kwargs):
        """Quick plot via MNE's ``raw.plot()``."""
        return self.raw.plot(**kwargs)

    # ------------------------------------------------------------------
    # Info / display
    # ------------------------------------------------------------------

    def info(self) -> dict:
        """Return a summary dict."""
        return {
            "filepath": self._filepath,
            "data_type": self.data_type,
            "sampling_frequency": self.sampling_frequency,
            "channel_count": self.channel_count,
            "duration": self.duration,
            "n_annotations": len(self.raw.annotations),
        }

    def __repr__(self) -> str:
        if self._raw is None:
            return "fNIRS(no data loaded)"
        return (
            f"fNIRS(type={self.data_type}, "
            f"channels={self.channel_count}, "
            f"fs={self.sampling_frequency} Hz, "
            f"duration={self.duration:.1f}s)"
        )

    def print(self) -> None:
        """Print summary info."""
        for k, v in self.info().items():
            print(f"  {k}: {v}")

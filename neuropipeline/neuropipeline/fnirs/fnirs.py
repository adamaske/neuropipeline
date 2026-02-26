"""
fNIRS processing class.

All file I/O uses h5py directly via SNIRFImporter / SNIRFExporter.
No MNE dependency.
"""

import os
import shutil

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

from .snirf_types import SNIRF
from .fnirs_data import fnirs_data_type, WL, OD, CC
from .preprocessor import TDDR, butter_bandpass_filter, fNIRSPreprocessor


# ============================================================================
# FFT / PSD helpers (unchanged)
# ============================================================================

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


# ============================================================================
# Beer-Lambert law helpers
# ============================================================================

# Molar extinction coefficients at selected wavelengths in cm⁻¹/M.
# Source: Matcher SJ et al., Applied Spectroscopy 49(1), 1995.
# HbO and HbR values at the same wavelength are (ε_HbO, ε_HbR).
_EXT_WAVELENGTHS = np.array([
    650,  660,  670,  680,  690,
    700,  710,  720,  730,  740,
    750,  760,  770,  780,  790,
    800,  808,  820,  830,  840,
    850,  860,  870,  880,  890,  900,
], dtype=float)

_EXT_HbO = np.array([
     607,  641,  666,  645,  698,
     733,  754,  812,  871, 1041,
    1053, 1506, 1274, 1427, 1613,
    1822, 2014, 2246, 2416, 2663,
    2761, 3114, 3233, 3262, 3118, 2869,
], dtype=float)

_EXT_HbR = np.array([
    2688, 2566, 2421, 2327, 2383,
    2167, 1901, 1679, 1467, 1304,
    1120, 3483,  869,  773,  728,
     700, 2021,  672,  667,  643,
    1735,  651,  658,  671,  703,  756,
], dtype=float)


def get_extinction_coefficients(wavelength_nm: float) -> tuple[float, float]:
    """
    Return (ε_HbO, ε_HbR) in cm⁻¹/M via linear interpolation.

    Source: Matcher et al. 1995, Applied Spectroscopy 49(1).
    """
    lo, hi = float(_EXT_WAVELENGTHS[0]), float(_EXT_WAVELENGTHS[-1])
    if wavelength_nm < lo or wavelength_nm > hi:
        print(f"Warning: wavelength {wavelength_nm} nm outside tabulated range "
              f"[{lo:.0f}, {hi:.0f}]. Extrapolation may be inaccurate.")
    e_hbo = float(np.interp(wavelength_nm, _EXT_WAVELENGTHS, _EXT_HbO))
    e_hbr = float(np.interp(wavelength_nm, _EXT_WAVELENGTHS, _EXT_HbR))
    return e_hbo, e_hbr


def _optode_distance_cm(src_optode, det_optode) -> float:
    """
    Compute source-detector Euclidean distance in cm.

    SNIRF probe positions are in metres (SI). Converts to cm for Beer-Lambert.
    Falls back to a typical 3 cm separation if positions are unavailable.
    """
    DEFAULT_DISTANCE_CM = 3.0

    pos_src = src_optode.position_3D or src_optode.position_2D
    pos_det = det_optode.position_3D or det_optode.position_2D
    if pos_src is None or pos_det is None:
        return DEFAULT_DISTANCE_CM

    dx = pos_src.x - pos_det.x
    dy = pos_src.y - pos_det.y
    dz = (pos_src.z - pos_det.z) if hasattr(pos_src, "z") and hasattr(pos_det, "z") else 0.0
    d_m = float(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
    d_cm = d_m * 100.0 if d_m < 1.0 else d_m  # assume metres if < 1, else cm
    return d_cm if d_cm > 0.0 else DEFAULT_DISTANCE_CM


# ============================================================================
# fNIRS processing class
# ============================================================================

class fNIRS:
    """
    Stateful fNIRS processing class.

    Loads data via the h5py-based SNIRFImporter; keeps the full SNIRF struct
    in self.snirf and maintains flat numpy arrays (self.channel_data, etc.)
    as the mutable processing state.
    """

    def __init__(self, filepath: str | None = None):
        self.snirf: SNIRF | None = None
        self.type = WL

        self.sampling_frequency: float | None = None
        self.channel_names: list[str] | None = None
        self.channel_data: np.ndarray | None = None   # (channels × samples)
        self.channel_num: int | None = None

        self.feature_onsets: np.ndarray | None = None
        self.feature_descriptions: np.ndarray | None = None
        self.feature_durations: np.ndarray | None = None

        self.channel_dict = None

        if filepath is not None:
            self.read_snirf(filepath)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def read_snirf(self, filepath: str) -> None:
        from .importers import SNIRFImporter
        self.snirf = SNIRFImporter().load(filepath)
        self.type = WL
        self._populate_flat_fields()

    def _populate_flat_fields(self) -> None:
        """Sync flat convenience fields from self.snirf."""
        s = self.snirf
        self.sampling_frequency = s.get_sampling_rate()
        self.channel_names = s.get_channel_names()
        self.channel_data = s.channel_store.data.copy()
        self.channel_num = self.channel_data.shape[0]

        # Flatten all events into parallel arrays, sorted by onset
        all_onsets: list[float] = []
        all_descs:  list[int]   = []
        all_durs:   list[float] = []

        for event in s.events.events:
            for marker in event.markers:
                all_onsets.append(marker.onset)
                all_durs.append(marker.duration)
                try:
                    all_descs.append(int(event.name))
                except (ValueError, TypeError):
                    all_descs.append(0)

        if all_onsets:
            idx = np.argsort(all_onsets)
            self.feature_onsets       = np.array(all_onsets)[idx]
            self.feature_descriptions = np.array(all_descs)[idx]
            self.feature_durations    = np.array(all_durs)[idx]
        else:
            self.feature_onsets       = np.array([], dtype=float)
            self.feature_descriptions = np.array([], dtype=int)
            self.feature_durations    = np.array([], dtype=float)

    def write_snirf(self, filepath: str) -> None:
        """Write the current processing state back to a SNIRF file."""
        if self.snirf is None:
            raise RuntimeError("No SNIRF data loaded.")

        stem, suffix = os.path.splitext(filepath)
        temp_path = stem + "_temporary" + suffix
        shutil.copy(self.snirf.filepath, temp_path)

        with h5py.File(temp_path, "r+") as f:
            nirs_keys = [k for k in f.keys() if k.startswith("nirs")]
            nirs_key = "nirs" if "nirs" in nirs_keys else sorted(nirs_keys)[0]
            nirs = f[nirs_key]

            # Overwrite dataTimeSeries and time
            if "data1" in nirs:
                dg = nirs["data1"]
                if "dataTimeSeries" in dg:
                    del dg["dataTimeSeries"]
                dg.create_dataset("dataTimeSeries",
                                  data=self.channel_data.T.astype(np.float64))
                if "time" in dg:
                    del dg["time"]
                dg.create_dataset("time", data=self.snirf.get_time().astype(np.float64))

            # Rebuild stim blocks from flat feature arrays
            for key in [k for k in nirs.keys() if k.startswith("stim")]:
                del nirs[key]

            if len(self.feature_onsets) > 0:
                unique_descs = sorted(set(self.feature_descriptions))
                for i, desc in enumerate(unique_descs, start=1):
                    mask = self.feature_descriptions == desc
                    onsets    = self.feature_onsets[mask]
                    durations = self.feature_durations[mask]
                    amplitudes = np.ones(len(onsets))
                    sg = nirs.create_group(f"stim{i}")
                    sg.create_dataset(
                        "data",
                        data=np.column_stack([onsets, durations, amplitudes]).astype(np.float64),
                    )
                    sg.create_dataset("name", data=str(desc).encode("utf-8"))

        if os.path.exists(filepath):
            ans = input(f"{filepath} already exists. Overwrite? [Y/N]: ")
            if ans.strip().upper() == "Y":
                shutil.move(temp_path, filepath)
                print(f"fNIRS.write_snirf: wrote {filepath}")
            else:
                os.remove(temp_path)
                print("fNIRS.write_snirf: cancelled.")
        else:
            os.rename(temp_path, filepath)
            print(f"fNIRS.write_snirf: wrote {filepath}")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> dict:
        """Return metadata tags as a dict {name: value}."""
        if self.snirf is None:
            raise RuntimeError("No SNIRF data loaded.")
        return {tag.name: tag.value for tag in self.snirf.metadata.tags}

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def print(self) -> None:
        print("fNIRS")
        print(f"  type               : {self.type.value}")
        print(f"  sampling_frequency : {self.sampling_frequency} Hz")
        print(f"  channel_num        : {self.channel_num}")
        print(f"  channel_data       : {self.channel_data.shape}")
        print(f"  channel_names      : {self.channel_names}")
        print(f"  feature_onsets     : {self.feature_onsets}")
        print(f"  feature_descriptions: {self.feature_descriptions}")

    def get_duration(self) -> float:
        return self.channel_data.shape[1] / self.sampling_frequency

    def get_time(self) -> np.ndarray:
        return np.arange(self.channel_data.shape[1]) / self.sampling_frequency

    # ------------------------------------------------------------------
    # Channel helpers
    # ------------------------------------------------------------------

    def get_channel_dict(self) -> dict:
        self.channel_dict = {}
        for i, channel_name in enumerate(self.channel_names):
            source_detector = channel_name.split()[0]
            wavelength = channel_name.split()[1]
            if source_detector not in self.channel_dict:
                self.channel_dict[source_detector] = {"HbO": None, "HbR": None}
            ch_data = self.channel_data[i]
            wl_lower = wavelength.lower()
            if wl_lower in ("hbr", "760"):
                self.channel_dict[source_detector]["HbR"] = ch_data
            if wl_lower in ("hbo", "850"):
                self.channel_dict[source_detector]["HbO"] = ch_data
        return self.channel_dict

    def split(self):
        """Split channel_data into HbO and HbR arrays. Returns (hbo, hbo_names, hbr, hbr_names)."""
        assert len(self.channel_data) == len(self.channel_names)
        hbo_channels, hbo_names = [], []
        hbr_channels, hbr_names = [], []
        for i, ch_name in enumerate(self.channel_names):
            parts = ch_name.split()
            assert len(parts) == 2, f"Unexpected channel name format: {ch_name!r}"
            source_detector = parts[0]
            wavelength = parts[1].lower()
            if wavelength in ("hbr", "760"):
                hbr_channels.append(self.channel_data[i])
                hbr_names.append(source_detector)
            if wavelength in ("hbo", "850"):
                hbo_channels.append(self.channel_data[i])
                hbo_names.append(source_detector)
        return np.array(hbo_channels), hbo_names, np.array(hbr_channels), hbr_names

    # ------------------------------------------------------------------
    # Downsampling
    # ------------------------------------------------------------------

    def downsample(self, factor: int) -> None:
        if factor <= 1:
            print("Downsample factor must be greater than 1.")
            return
        self.channel_data = self.channel_data[:, ::factor]
        self.sampling_frequency /= factor
        print(f"Downsampled by {factor}×. New fs: {self.sampling_frequency} Hz")

    # ------------------------------------------------------------------
    # Preprocessing steps
    # ------------------------------------------------------------------

    def to_optical_density(self, use_initial_value: bool = False) -> None:
        """Convert raw light intensity to optical density."""
        if self.type != WL:
            print(f"Type is {self.type}, cannot convert to {OD}.")
            return

        if use_initial_value:
            safe = np.clip(self.channel_data, a_min=1e-12, a_max=None)
            safe_init = np.clip(self.channel_data[:, 0:1], a_min=1e-12, a_max=None)
            od = -np.log(safe / safe_init)
        else:
            data = np.abs(self.channel_data)
            min_nz = np.min(np.where(data > 0, data, np.inf), axis=1, keepdims=True)
            data = np.maximum(data, min_nz)
            means = np.mean(data, axis=1, keepdims=True)
            od = -np.log(data / means)

        self.channel_data = od
        self.type = OD

    def to_hemoglobin_concentration(self, dpf: float = 6.0) -> None:
        """
        Convert optical density to haemoglobin concentration via the modified
        Beer-Lambert law.

        Requires two wavelengths to be present in probe.wavelengths.
        Channel layout assumed (mirrors ParseData1): first N/2 rows are at the
        lower wavelength (HbR-dominant), second N/2 rows at the higher
        wavelength (HbO-dominant).

        Args:
            dpf: Differential pathlength factor (dimensionless).
                 Default 6.0 is a typical value for adult brain tissue.

        Extinction coefficients:
            Source — Matcher SJ et al., Applied Spectroscopy 49(1), 1995.
            Units  — cm⁻¹/M. Distance derived from probe positions (metres,
                     converted to cm); falls back to 3 cm if unavailable.

        Output units: mol/L (M), consistent with MNE's beer_lambert_law.
        """
        if self.type != OD:
            print(f"Type is {self.type}, cannot convert to {CC}.")
            return

        wl = self.snirf.probe.wavelengths if self.snirf else []
        if len(wl) < 2:
            raise ValueError(
                "Need at least 2 wavelengths in probe.wavelengths for Beer-Lambert."
            )

        wl_lo, wl_hi = sorted(wl)[:2]
        e_hbo_lo, e_hbr_lo = get_extinction_coefficients(wl_lo)
        e_hbo_hi, e_hbr_hi = get_extinction_coefficients(wl_hi)

        # Extinction matrix [cm⁻¹/M]
        E = np.array([[e_hbo_lo, e_hbr_lo],
                      [e_hbo_hi, e_hbr_hi]])
        E_inv = np.linalg.inv(E)

        n_total = self.channel_data.shape[0]
        if n_total % 2 != 0:
            raise ValueError(
                f"Expected an even number of channels for Beer-Lambert ({n_total} found)."
            )
        n_pairs = n_total // 2

        new_data  = np.zeros_like(self.channel_data)
        new_names = [""] * n_total

        channels_map = self.snirf.probe.channels if self.snirf else {}

        for i in range(n_pairs):
            od_lo = self.channel_data[i]            # lower wavelength (HbR row)
            od_hi = self.channel_data[i + n_pairs]  # higher wavelength (HbO row)

            # Source-detector distance → cm
            d_cm = 3.0
            if self.snirf and i in channels_map:
                ch = channels_map[i]
                src = self.snirf.probe.sources.get(ch.source_id)
                det = self.snirf.probe.detectors.get(ch.detector_id)
                if src and det:
                    d_cm = _optode_distance_cm(src, det)

            # [2, n_samples] = E_inv · [OD_lo, OD_hi] / (dpf * d_cm)
            # Output in mol/L
            od_pair = np.vstack([od_lo, od_hi])
            dC = E_inv @ od_pair / (dpf * d_cm)   # (2, n_samples) in mol/L

            # First n_pairs rows = HbR, next n_pairs rows = HbO
            new_data[i]           = dC[1]   # HbR
            new_data[i + n_pairs] = dC[0]   # HbO

            sd = f"S{channels_map[i].source_id}-D{channels_map[i].detector_id}" \
                if i in channels_map else f"ch{i}"
            new_names[i]           = f"{sd} hbr"
            new_names[i + n_pairs] = f"{sd} hbo"

        self.channel_data  = new_data
        self.channel_names = new_names
        self.type          = CC

    def bandpass_channels(self, low_freq: float = 0.01,
                          high_freq: float = 0.1, order: int = 5) -> None:
        for i, ch in enumerate(self.channel_data):
            self.channel_data[i] = butter_bandpass_filter(
                ch, low_freq, high_freq, self.sampling_frequency, order
            )

    def preprocess(self,
                   preprocessor: fNIRSPreprocessor = None,
                   optical_density: bool = True,
                   hemoglobin_concentration: bool = True,
                   motion_correction: bool = True,
                   temporal_filtering: bool = True,
                   detrending: bool = True,
                   normalization: bool = True) -> None:
        if preprocessor is not None:
            pp = preprocessor
        else:
            pp = fNIRSPreprocessor(
                optical_density=optical_density,
                hemoglobin_concentration=hemoglobin_concentration,
                motion_correction=motion_correction,
                temporal_filtering=temporal_filtering,
                detrending=detrending,
                normalization=normalization,
            )

        if pp.optical_density:
            self.to_optical_density(use_initial_value=pp.od_use_initial_value)

        if pp.motion_correction:
            for i, ch in enumerate(self.channel_data):
                self.channel_data[i] = TDDR(ch, self.sampling_frequency)

        if pp.hemoglobin_concentration:
            self.to_hemoglobin_concentration()

        if pp.temporal_filtering:
            self.bandpass_channels(pp.bandpass_lowcut, pp.bandpass_highcut,
                                   pp.bandpass_order)

        if pp.detrending:
            for i, ch in enumerate(self.channel_data):
                self.channel_data[i] = detrend(ch, type=pp.detrend_type)

        if pp.normalization:
            means = np.mean(self.channel_data, axis=1, keepdims=True)
            stds  = np.std(self.channel_data, axis=1, keepdims=True)
            stds[stds == 0] = 1
            self.channel_data = (self.channel_data - means) / stds

    # ------------------------------------------------------------------
    # Trimming
    # ------------------------------------------------------------------

    def trim(self, start_seconds: float = 0.0, end_seconds: float = 0.0) -> None:
        """Trim start_seconds from the beginning and end_seconds from the end."""
        total = self.channel_data.shape[1]
        duration = total / self.sampling_frequency

        if start_seconds + end_seconds >= duration:
            raise ValueError(
                f"Cannot trim {start_seconds}s + {end_seconds}s from a "
                f"{duration:.2f}s recording."
            )

        start_sample = int(round(start_seconds * self.sampling_frequency))
        end_sample   = total - int(round(end_seconds * self.sampling_frequency))

        print(f"trim: [{start_seconds}s : {duration - end_seconds:.2f}s] "
              f"→ {end_sample - start_sample} samples")

        self.channel_data = self.channel_data[:, start_sample:end_sample]

        # Adjust features
        valid = (self.feature_onsets >= start_seconds) & \
                (self.feature_onsets < duration - end_seconds)
        self.feature_onsets       = self.feature_onsets[valid] - start_seconds
        self.feature_descriptions = self.feature_descriptions[valid]
        self.feature_durations    = self.feature_durations[valid]

    def trim_from_features(self, cut_from_first: float = 5.0,
                           cut_from_last: float = 10.0) -> None:
        first_s = self.feature_onsets[0]
        last_s  = self.feature_onsets[-1]
        self.trim(start_seconds=max(0.0, first_s - cut_from_first),
                  end_seconds=max(0.0,
                      self.channel_data.shape[1] / self.sampling_frequency
                      - last_s - cut_from_last))

    # ------------------------------------------------------------------
    # Feature manipulation
    # ------------------------------------------------------------------

    def remove_features(self, features_to_remove) -> None:
        keep = [i for i, d in enumerate(self.feature_descriptions)
                if d not in features_to_remove]
        n_removed = len(self.feature_descriptions) - len(keep)
        self.feature_onsets       = self.feature_onsets[keep]
        self.feature_descriptions = self.feature_descriptions[keep]
        self.feature_durations    = self.feature_durations[keep]
        print(f"Removed {n_removed} features. Remaining: {len(self.feature_descriptions)}.")

    def add_features(self,
                     onsets,
                     descriptions,
                     durations=0.0,
                     sort: bool = True) -> None:
        onsets       = np.array(onsets, dtype=float)
        descriptions = np.array(descriptions)
        if np.isscalar(durations):
            durations = np.full(len(onsets), float(durations))
        else:
            durations = np.array(durations, dtype=float)

        if len(onsets) != len(descriptions):
            raise ValueError("onsets and descriptions must have the same length.")

        self.feature_onsets       = np.concatenate([self.feature_onsets, onsets])
        self.feature_descriptions = np.concatenate([self.feature_descriptions, descriptions])
        self.feature_durations    = np.concatenate([self.feature_durations, durations])

        if sort:
            idx = np.argsort(self.feature_onsets)
            self.feature_onsets       = self.feature_onsets[idx]
            self.feature_descriptions = self.feature_descriptions[idx]
            self.feature_durations    = self.feature_durations[idx]

        print(f"Added {len(onsets)} features. Total: {len(self.feature_onsets)}.")

    def replace_features(self, onsets=None, descriptions=None,
                         durations=None) -> None:
        ref_length = (len(np.asarray(onsets)) if onsets is not None
                      else len(np.asarray(descriptions)) if descriptions is not None
                      else len(self.feature_onsets))

        if onsets is not None:
            self.feature_onsets = np.array(onsets, dtype=float)
        if descriptions is not None:
            self.feature_descriptions = np.array(descriptions)
        if durations is not None:
            if np.isscalar(durations):
                self.feature_durations = np.full(ref_length, float(durations))
            else:
                self.feature_durations = np.array(durations, dtype=float)

        print(f"Replaced features: {ref_length} markers set.")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_channels(self) -> None:
        hbo_data, hbo_names, hbr_data, hbr_names = self.split()

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        for i, ch in enumerate(hbo_data):
            plt.plot(ch, label=hbo_names[i])
        plt.title("HbO Time Series")
        plt.legend()

        plt.subplot(2, 2, 2)
        for i, ch in enumerate(hbr_data):
            plt.plot(ch, label=hbr_names[i])
        plt.title("HbR Time Series")
        plt.legend()

        plt.subplot(2, 2, 3)
        for ch in hbo_data:
            freqs, spectra = compute_psd(ch, self.sampling_frequency,
                                         int(self.sampling_frequency / 2))
            plt.plot(freqs, spectra)
        plt.title("HbO : Power Spectral Density")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD [V²/Hz]")

        plt.subplot(2, 2, 4)
        for ch in hbr_data:
            freqs, spectra = compute_psd(ch, self.sampling_frequency,
                                         int(self.sampling_frequency / 2))
            plt.plot(freqs, spectra)
        plt.title("HbR : Power Spectral Density")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD [V²/Hz]")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Epoch extraction (stub — unchanged from original)
    # ------------------------------------------------------------------

    def feature_epochs(self, feature_description, tmin, tmax):
        onsets = [self.feature_onsets[i]
                  for i, d in enumerate(self.feature_descriptions)
                  if d == feature_description]
        print(f"feature {feature_description}: {len(onsets)} epochs")
        print("onsets:", onsets)

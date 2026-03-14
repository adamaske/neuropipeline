"""Tests for the MNE-NIRS wrapper fNIRS module."""

import os
import tempfile

import mne
import numpy as np
import pytest

from neuropipeline.fnirs import fNIRS, fNIRSPreprocessor, compute_fft, compute_psd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_fnirs(n_sources=2, n_detectors=1, sfreq=10.0, duration_s=100):
    """Create a synthetic fNIRS Raw object with proper wavelength metadata."""
    wavelengths = [760.0, 850.0]
    ch_names = []
    for s in range(1, n_sources + 1):
        for d in range(1, n_detectors + 1):
            for wl in wavelengths:
                ch_names.append(f"S{s}_D{d} {int(wl)}")

    n_channels = len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="fnirs_cw_amplitude")

    for i, ch in enumerate(info["chs"]):
        wl = float(ch_names[i].split()[-1])
        parts = ch_names[i].split("_")
        src_id = int(parts[0][1:]) - 1
        det_id = int(parts[1].split()[0][1:]) - 1

        ch["loc"][3] = src_id * 0.03
        ch["loc"][4] = 0.0
        ch["loc"][5] = 0.0
        ch["loc"][6] = det_id * 0.03 + 0.03
        ch["loc"][7] = 0.0
        ch["loc"][8] = 0.0
        ch["loc"][9] = wl

    n_samples = int(sfreq * duration_s)
    data = np.abs(np.random.RandomState(42).rand(n_channels, n_samples)) * 1e-6 + 1e-6
    return mne.io.RawArray(data, info)


def _make_fif_file(tmp_path):
    """Write a temporary FIF file and return its path."""
    raw = _make_raw_fnirs()
    filepath = os.path.join(str(tmp_path), "test_raw.fif")
    raw.save(filepath, overwrite=True)
    return filepath


def _make_fnirs_from_raw():
    """Create an fNIRS wrapper with synthetic data (no file I/O)."""
    f = fNIRS()
    f._raw = _make_raw_fnirs()
    return f


# ---------------------------------------------------------------------------
# Tests — fNIRS class
# ---------------------------------------------------------------------------

class TestFNIRSProperties:

    def test_no_data_raises(self):
        f = fNIRS()
        with pytest.raises(RuntimeError, match="No data loaded"):
            _ = f.raw

    def test_properties(self):
        f = _make_fnirs_from_raw()
        assert f.sampling_frequency == 10.0
        assert f.channel_count == 4
        assert f.duration > 0
        assert isinstance(f.channel_names, list)
        assert len(f.channel_names) == 4

    def test_data_type_raw(self):
        f = _make_fnirs_from_raw()
        assert f.data_type == "raw"

    def test_repr(self):
        f = _make_fnirs_from_raw()
        r = repr(f)
        assert "fNIRS" in r
        assert "channels=4" in r

    def test_repr_no_data(self):
        f = fNIRS()
        assert "no data loaded" in repr(f)

    def test_info_dict(self):
        f = _make_fnirs_from_raw()
        d = f.info()
        assert "sampling_frequency" in d
        assert "channel_count" in d
        assert d["channel_count"] == 4


class TestFNIRSPreprocessing:

    def test_to_optical_density(self):
        f = _make_fnirs_from_raw()
        result = f.to_optical_density()
        assert result is f  # chaining
        assert f.data_type == "optical_density"

    def test_to_hemoglobin(self):
        f = _make_fnirs_from_raw()
        f.to_optical_density()
        result = f.to_hemoglobin()
        assert result is f
        assert f.data_type == "hemoglobin"

    def test_chaining(self):
        f = _make_fnirs_from_raw()
        result = f.to_optical_density().to_hemoglobin().bandpass(0.01, 4.0)
        assert result is f
        assert f.data_type == "hemoglobin"

    def test_bandpass(self):
        f = _make_fnirs_from_raw()
        f.to_optical_density().to_hemoglobin()
        result = f.bandpass(0.01, 4.0)
        assert result is f

    def test_crop(self):
        f = _make_fnirs_from_raw()
        original_duration = f.duration
        f.crop(tmin=10.0, tmax=50.0)
        assert f.duration < original_duration
        assert f.duration == pytest.approx(40.0, abs=0.2)

    def test_resample(self):
        f = _make_fnirs_from_raw()
        f.resample(5.0)
        assert f.sampling_frequency == 5.0

    def test_get_hbo_hbr(self):
        f = _make_fnirs_from_raw()
        f.to_optical_density().to_hemoglobin()
        hbo_data, hbo_names = f.get_hbo()
        hbr_data, hbr_names = f.get_hbr()
        assert hbo_data.shape[0] > 0
        assert hbr_data.shape[0] > 0
        assert len(hbo_names) == hbo_data.shape[0]
        assert len(hbr_names) == hbr_data.shape[0]

    def test_get_data(self):
        f = _make_fnirs_from_raw()
        data = f.get_data()
        assert data.shape == (4, 1000)

    def test_raw_escape_hatch(self):
        f = _make_fnirs_from_raw()
        assert isinstance(f.raw, mne.io.BaseRaw)

    def test_preprocess_with_kwargs(self):
        f = _make_fnirs_from_raw()
        f.preprocess(motion_correction=False, bandpass=False)
        assert f.data_type == "hemoglobin"


class TestFNIRSIO:

    def test_load_fif(self, tmp_path):
        filepath = _make_fif_file(tmp_path)
        f = fNIRS(filepath)
        assert f.channel_count == 4
        assert f.sampling_frequency == 10.0

    def test_save_fif(self, tmp_path):
        f = _make_fnirs_from_raw()
        out_path = os.path.join(str(tmp_path), "output_raw.fif")
        f.save(out_path)
        assert os.path.exists(out_path)
        # Verify round-trip
        f2 = fNIRS(out_path)
        assert f2.channel_count == f.channel_count

    def test_csv_export(self, tmp_path):
        f = _make_fnirs_from_raw()
        f.to_optical_density().to_hemoglobin()
        out_dir = os.path.join(str(tmp_path), "csv_out")
        f.to_csv(out_dir, name="test")
        assert os.path.exists(os.path.join(out_dir, "test_channels.csv"))
        assert os.path.exists(os.path.join(out_dir, "test_events.csv"))

        # Verify CSV content
        import pandas as pd
        df = pd.read_csv(os.path.join(out_dir, "test_channels.csv"))
        assert "time" in df.columns
        assert len(df.columns) == f.channel_count + 1  # time + channels


class TestFNIRSEvents:

    def test_annotations_empty(self):
        f = _make_fnirs_from_raw()
        assert len(f.annotations) == 0


# ---------------------------------------------------------------------------
# Tests — fNIRSPreprocessor
# ---------------------------------------------------------------------------

class TestPreprocessor:

    def test_default_pipeline(self):
        f = _make_fnirs_from_raw()
        pp = fNIRSPreprocessor()
        pp.apply(f)
        assert f.data_type == "hemoglobin"

    def test_builder_pattern(self):
        pp = fNIRSPreprocessor()
        result = pp.set_motion_correction(False)
        assert result is pp

    def test_selective_steps(self):
        f = _make_fnirs_from_raw()
        pp = fNIRSPreprocessor(
            optical_density=True,
            motion_correction=False,
            hemoglobin=True,
            bandpass=False,
        )
        pp.apply(f)
        assert f.data_type == "hemoglobin"

    def test_od_only(self):
        f = _make_fnirs_from_raw()
        pp = fNIRSPreprocessor(
            optical_density=True,
            motion_correction=False,
            hemoglobin=False,
            bandpass=False,
        )
        pp.apply(f)
        assert f.data_type == "optical_density"

    def test_repr(self):
        pp = fNIRSPreprocessor()
        r = repr(pp)
        assert "OD" in r
        assert "TDDR" in r
        assert "HbX" in r

    def test_print(self, capsys):
        pp = fNIRSPreprocessor()
        pp.print()
        captured = capsys.readouterr()
        assert "optical_density" in captured.out
        assert "motion_correction" in captured.out


# ---------------------------------------------------------------------------
# Tests — FFT / PSD helpers
# ---------------------------------------------------------------------------

class TestFFTPSD:

    def test_compute_fft(self):
        fs = 100.0
        t = np.arange(0, 1.0, 1.0 / fs)
        signal = np.sin(2 * np.pi * 10 * t)
        freqs, spectrum = compute_fft(signal, fs, freq_limit=50.0)
        assert len(freqs) == len(spectrum)
        # Peak should be near 10 Hz
        peak_freq = freqs[np.argmax(spectrum)]
        assert abs(peak_freq - 10.0) < 2.0

    def test_compute_psd(self):
        fs = 100.0
        t = np.arange(0, 1.0, 1.0 / fs)
        signal = np.sin(2 * np.pi * 10 * t)
        freqs, psd = compute_psd(signal, fs, freq_limit=50.0)
        assert len(freqs) == len(psd)
        assert np.all(psd >= 0)

    def test_fft_no_limit(self):
        signal = np.random.rand(100)
        freqs, spectrum = compute_fft(signal, 100.0, freq_limit=None)
        assert len(freqs) == 50

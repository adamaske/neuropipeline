"""
Preprocessing utilities for fNIRS data.

Contains the fNIRSPreprocessor class for configuring preprocessing pipelines,
and signal processing functions (TDDR, bandpass filters, etc.).
"""
import numpy as np
from scipy.signal import butter, sosfreqz, sosfiltfilt, iirnotch, filtfilt, detrend


def TDDR(signal, sample_rate):
    """
    Temporal Derivative Distribution Repair (TDDR) motion correction.

    Reference implementation for the TDDR algorithm for motion correction
    of fNIRS data, as described in:

    Fishburn F.A., Ludlum R.S., Vaidya C.J., & Medvedev A.V. (2019).
    Temporal Derivative Distribution Repair (TDDR): A motion correction
    method for fNIRS. NeuroImage, 184, 171-179.
    https://doi.org/10.1016/j.neuroimage.2018.09.025

    Args:
        signal: A [sample x channel] matrix of uncorrected optical density data
        sample_rate: A scalar reflecting the rate of acquisition in Hz

    Returns:
        signals_corrected: A [sample x channel] matrix of corrected optical density data
    """
    signal = np.array(signal)
    if len(signal.shape) != 1:
        for ch in range(signal.shape[1]):
            signal[:, ch] = TDDR(signal[:, ch], sample_rate)
        return signal

    # Preprocess: Separate high and low frequencies
    filter_cutoff = .5
    filter_order = 3
    Fc = filter_cutoff * 2 / sample_rate
    signal_mean = np.mean(signal)
    signal -= signal_mean
    if Fc < 1:
        fb, fa = butter(filter_order, Fc)
        signal_low = filtfilt(fb, fa, signal, padlen=0)
    else:
        signal_low = signal

    signal_high = signal - signal_low

    # Initialize
    tune = 4.685
    D = np.sqrt(np.finfo(signal.dtype).eps)
    mu = np.inf
    iter = 0

    # Step 1. Compute temporal derivative of the signal
    deriv = np.diff(signal_low)

    # Step 2. Initialize observation weights
    w = np.ones(deriv.shape)

    # Step 3. Iterative estimation of robust weights
    while iter < 50:
        iter = iter + 1
        mu0 = mu

        # Step 3a. Estimate weighted mean
        mu = np.sum(w * deriv) / np.sum(w)

        # Step 3b. Calculate absolute residuals of estimate
        dev = np.abs(deriv - mu)

        # Step 3c. Robust estimate of standard deviation of the residuals
        sigma = 1.4826 * np.median(dev)

        # Step 3d. Scale deviations by standard deviation and tuning parameter
        r = dev / (sigma * tune)

        # Step 3e. Calculate new weights according to Tukey's biweight function
        w = ((1 - r**2) * (r < 1)) ** 2

        # Step 3f. Terminate if new estimate is within machine-precision of old estimate
        if abs(mu - mu0) < D * max(abs(mu), abs(mu0)):
            break

    # Step 4. Apply robust weights to centered derivative
    new_deriv = w * (deriv - mu)

    # Step 5. Integrate corrected derivative
    signal_low_corrected = np.cumsum(np.insert(new_deriv, 0, 0.0))

    # Postprocess: Center the corrected signal
    signal_low_corrected = signal_low_corrected - np.mean(signal_low_corrected)

    # Postprocess: Merge back with uncorrected high frequency component
    signal_corrected = signal_low_corrected + signal_high + signal_mean

    return signal_corrected


def butter_bandpass(lowcut, highcut, fs, freqs=512, order=3):
    """Design a butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    w, h = sosfreqz(sos, worN=None, whole=True, fs=fs)
    return sos, w, h


def butter_bandpass_filter(time_series, lowcut, highcut, fs, order):
    """Apply a butterworth bandpass filter to a time series."""
    sos, w, h = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, time_series)
    return np.array(y)


def notch_filter(data, sfreq, freqs=[50, 60]):
    """Apply notch filters at specified frequencies (e.g., powerline noise)."""
    for freq in freqs:
        b, a = iirnotch(freq, 30, sfreq)
        data = filtfilt(b, a, data, axis=-1)
    return data


class fNIRSPreprocessor:
    """
    Configurable preprocessor for fNIRS data.

    Usage:
        # Create with defaults
        preprocessor = fNIRSPreprocessor()

        # Customize settings
        preprocessor.set_bandpass(0.01, 0.2, order=10)
        preprocessor.motion_correction = False

        # Apply to data
        fnirs.preprocess(preprocessor)
    """

    def __init__(self,
                 optical_density: bool = True,
                 hemoglobin_concentration: bool = True,
                 motion_correction: bool = True,
                 temporal_filtering: bool = True,
                 detrending: bool = True,
                 normalization: bool = True):

        # Pipeline steps (on/off)
        self.optical_density = optical_density
        self.hemoglobin_concentration = hemoglobin_concentration
        self.motion_correction = motion_correction
        self.temporal_filtering = temporal_filtering
        self.detrending = detrending
        self.normalization = normalization

        # Optical density settings
        self.od_use_initial_value = False

        # Bandpass filter settings
        self.bandpass_lowcut = 0.01
        self.bandpass_highcut = 0.1
        self.bandpass_order = 15

        # Detrending settings
        self.detrend_type = 'linear'  # 'linear' or 'constant'

    def set_optical_density(self, enabled: bool = True, use_initial_value: bool = False):
        """
        Configure optical density conversion.

        Args:
            enabled: Whether to apply optical density conversion.
            use_initial_value: If True, use initial intensity value (Dan 2019).
                               If False, use mean (MNE default).
        """
        self.optical_density = enabled
        self.od_use_initial_value = use_initial_value
        return self

    def set_hemoglobin_concentration(self, enabled: bool = True):
        """
        Configure hemoglobin concentration conversion (Beer-Lambert law).

        Args:
            enabled: Whether to convert to hemoglobin concentration.
        """
        self.hemoglobin_concentration = enabled
        return self

    def set_motion_correction(self, enabled: bool = True):
        """
        Configure TDDR motion correction.

        Args:
            enabled: Whether to apply TDDR motion correction.
        """
        self.motion_correction = enabled
        return self

    def set_temporal_filtering(self, enabled: bool = True, lowcut: float = 0.01,
                                highcut: float = 0.1, order: int = 15):
        """
        Configure bandpass temporal filtering.

        Args:
            enabled: Whether to apply temporal filtering.
            lowcut: Low cutoff frequency in Hz.
            highcut: High cutoff frequency in Hz.
            order: Filter order.
        """
        self.temporal_filtering = enabled
        self.bandpass_lowcut = lowcut
        self.bandpass_highcut = highcut
        self.bandpass_order = order
        return self

    def set_detrending(self, enabled: bool = True, detrend_type: str = 'linear'):
        """
        Configure detrending.

        Args:
            enabled: Whether to apply detrending.
            detrend_type: Type of detrending ('linear' or 'constant').
        """
        self.detrending = enabled
        self.detrend_type = detrend_type
        return self

    def set_normalization(self, enabled: bool = True):
        """
        Configure z-score normalization.

        Args:
            enabled: Whether to apply z-score normalization.
        """
        self.normalization = enabled
        return self

    def print(self):
        """Print current preprocessor settings."""
        print("fNIRSPreprocessor Settings:")
        print(f"  optical_density: {self.optical_density} (use_initial_value={self.od_use_initial_value})")
        print(f"  motion_correction: {self.motion_correction}")
        print(f"  hemoglobin_concentration: {self.hemoglobin_concentration}")
        print(f"  temporal_filtering: {self.temporal_filtering} (lowcut={self.bandpass_lowcut}, highcut={self.bandpass_highcut}, order={self.bandpass_order})")
        print(f"  detrending: {self.detrending} (type={self.detrend_type})")
        print(f"  normalization: {self.normalization}")

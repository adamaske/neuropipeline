import numpy as np
import matplotlib.pyplot as plt
from .fnirs import fNIRS
import pywt


def complex_morlet_transform(signal, scales, wavelet='cmor1.5-1.0'):
    """
    Perform a complex Morlet wavelet transform on the input signal.

    Parameters:
    -----------
    signal : array-like
        The input signal to be transformed.
    scales : array-like
        The scales at which to compute the wavelet transform.
    wavelet : str
        The type of wavelet to use. Default is 'cmor1.5-1.0' (bandwidth-center frequency).
        Format: 'cmor{bandwidth}-{center_frequency}'

    Returns:
    --------
    tuple: (coefficients, frequencies)
        coefficients: Complex wavelet coefficients (n_scales, n_times)
        frequencies: Corresponding frequencies for each scale
    """
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
    return coefficients, frequencies

def cmt(signal, fs, freqs, n_cycles=5):
    """
    Compute Complex Morlet Transform at specific frequencies.

    Parameters:
    -----------
    signal : array-like
        Input signal
    fs : float
        Sampling frequency in Hz
    freqs : array-like
        Frequencies of interest in Hz
    n_cycles : float
        Number of cycles in the wavelet (controls time-frequency tradeoff)
        Higher values = better frequency resolution, worse time resolution

    Returns:
    --------
    coefficients : ndarray
        Complex coefficients (n_freqs, n_times)
    """
    # For complex Morlet: bandwidth parameter relates to n_cycles
    # bandwidth = 2 * n_cycles^2 / (pi * f^2) approximately
    # We use a simplified approach: scale bandwidth with n_cycles
    bandwidth = n_cycles / 2.0
    center_freq = 1.0

    wavelet = f'cmor{bandwidth:.1f}-{center_freq}'

    # Convert frequencies to scales: scale = (center_freq * fs) / freq
    scales = (center_freq * fs) / np.array(freqs)

    coefficients, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)

    return coefficients


def cmt_spectrogram(signal, fs, freq_range, n_freqs=50, n_cycles=5):
    """
    Compute a spectrogram using Complex Morlet Transform.

    Parameters:
    -----------
    signal : array-like
        Input signal
    fs : float
        Sampling frequency in Hz
    freq_range : tuple
        (min_freq, max_freq) in Hz
    n_freqs : int
        Number of frequency bins
    n_cycles : float
        Number of cycles in the wavelet

    Returns:
    --------
    dict with keys:
        'power': Power spectrogram (n_freqs, n_times)
        'phase': Phase spectrogram (n_freqs, n_times)
        'coefficients': Complex coefficients (n_freqs, n_times)
        'freqs': Frequency array
        'times': Time array
    """
    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
    # Avoid zero frequency
    freqs = freqs[freqs > 0]

    coefficients = cmt(signal, fs, freqs, n_cycles)

    times = np.arange(len(signal)) / fs

    return {
        'power': np.abs(coefficients) ** 2,
        'phase': np.angle(coefficients),
        'coefficients': coefficients,
        'freqs': freqs,
        'times': times
    }


def cmt_parameter_sweep(signal, fs, freq_range, cycle_range=(3, 15), n_cycles_steps=5):
    """
    Compare CMT outputs across different cycle parameters.

    Parameters:
    -----------
    signal : array-like
        Input signal
    fs : float
        Sampling frequency in Hz
    freq_range : tuple
        (min_freq, max_freq) in Hz
    cycle_range : tuple
        (min_cycles, max_cycles) to sweep
    n_cycles_steps : int
        Number of cycle values to test

    Returns:
    --------
    dict: Spectrograms keyed by n_cycles value
    """
    results = {}
    for n_cycles in np.linspace(cycle_range[0], cycle_range[1], n_cycles_steps):
        results[n_cycles] = cmt_spectrogram(signal, fs, freq_range, n_cycles=n_cycles)
    return results


def plot_spectrogram(spec_result, title='CMT Spectrogram', vmin=None, vmax=None, ax=None):
    """
    Plot a spectrogram result from cmt_spectrogram.

    Parameters:
    -----------
    spec_result : dict
        Output from cmt_spectrogram
    title : str
        Plot title
    vmin, vmax : float
        Color scale limits
    ax : matplotlib axis or None
        Axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    power = spec_result['power']
    times = spec_result['times']
    freqs = spec_result['freqs']

    im = ax.pcolormesh(times, freqs, power, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Power')

    return ax


def plot_parameter_comparison(sweep_results, signal, fs, figsize=(14, 10)):
    """
    Plot comparison of spectrograms with different n_cycles parameters.

    Parameters:
    -----------
    sweep_results : dict
        Output from cmt_parameter_sweep
    signal : array-like
        Original signal for reference
    fs : float
        Sampling frequency
    figsize : tuple
        Figure size
    """
    n_params = len(sweep_results)
    fig, axes = plt.subplots(n_params + 1, 1, figsize=figsize)

    # Plot original signal
    times = np.arange(len(signal)) / fs
    axes[0].plot(times, signal, 'k-', linewidth=0.5)
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original Signal')
    axes[0].set_xlim([times[0], times[-1]])

    # Find global min/max for consistent color scaling
    all_powers = [r['power'] for r in sweep_results.values()]
    vmax = max(p.max() for p in all_powers)

    # Plot each spectrogram
    for idx, (n_cycles, result) in enumerate(sorted(sweep_results.items())):
        ax = axes[idx + 1]
        im = ax.pcolormesh(result['times'], result['freqs'], result['power'],
                          shading='auto', cmap='jet', vmax=vmax)
        ax.set_ylabel('Freq [Hz]')
        ax.set_title(f'n_cycles = {n_cycles:.1f}')
        if idx == len(sweep_results) - 1:
            ax.set_xlabel('Time [s]')
        plt.colorbar(im, ax=ax, label='Power')

    plt.tight_layout()
    return fig, axes


# =============================================================================
# Example usage and demonstrations
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Complex Morlet Transform Analysis - Demo")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Basic CMT Spectrogram
    # -------------------------------------------------------------------------
    print("\n1. Basic CMT Spectrogram")
    print("-" * 40)

    # Create test signal: two sinusoids + chirp
    fs = 200  # Sampling frequency
    duration = 5  # seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # 7 Hz and 13 Hz components + a chirp from 20-40 Hz
    signal = (np.sin(2 * np.pi * 7 * t) +
              0.5 * np.sin(2 * np.pi * 13 * t) +
              0.3 * np.sin(2 * np.pi * (20 + 4*t) * t))  # chirp

    # Add some noise
    signal += 0.2 * np.random.randn(len(signal))

    # Compute spectrogram
    spec = cmt_spectrogram(signal, fs, freq_range=(1, 50), n_freqs=60, n_cycles=5)

    print(f"Signal length: {len(signal)} samples ({duration}s at {fs}Hz)")
    print(f"Spectrogram shape: {spec['power'].shape} (freqs x times)")
    print(f"Frequency range: {spec['freqs'].min():.1f} - {spec['freqs'].max():.1f} Hz")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].plot(t, signal, 'k-', linewidth=0.5)
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Test Signal: 7Hz + 13Hz + Chirp(20-40Hz) + Noise')
    axes[0].set_xlim([0, duration])

    plot_spectrogram(spec, title='CMT Spectrogram (n_cycles=5)', ax=axes[1])

    plt.tight_layout()
    plt.savefig('demo_1_basic_spectrogram.png', dpi=150)
    plt.show()

    # -------------------------------------------------------------------------
    # 2. Parameter Sweep - Effect of n_cycles
    # -------------------------------------------------------------------------
    print("\n2. Parameter Sweep - Effect of n_cycles")
    print("-" * 40)

    # Create a signal with transient events (good for showing time-freq tradeoff)
    t2 = np.linspace(0, 3, int(fs * 3), endpoint=False)
    signal2 = np.zeros_like(t2)

    # Add bursts at different times and frequencies
    # 10 Hz burst at t=0.5s
    burst_mask = (t2 > 0.4) & (t2 < 0.8)
    signal2[burst_mask] += np.sin(2 * np.pi * 10 * t2[burst_mask]) * np.hanning(burst_mask.sum())

    # 25 Hz burst at t=1.5s
    burst_mask = (t2 > 1.3) & (t2 < 1.7)
    signal2[burst_mask] += np.sin(2 * np.pi * 25 * t2[burst_mask]) * np.hanning(burst_mask.sum())

    # 40 Hz burst at t=2.5s
    burst_mask = (t2 > 2.3) & (t2 < 2.7)
    signal2[burst_mask] += np.sin(2 * np.pi * 40 * t2[burst_mask]) * np.hanning(burst_mask.sum())

    signal2 += 0.1 * np.random.randn(len(signal2))

    # Sweep parameters
    sweep_results = cmt_parameter_sweep(signal2, fs, freq_range=(1, 50),
                                        cycle_range=(2, 12), n_cycles_steps=4)

    print(f"Tested n_cycles values: {sorted(sweep_results.keys())}")

    fig, axes = plot_parameter_comparison(sweep_results, signal2, fs, figsize=(14, 12))
    plt.savefig('demo_2_parameter_sweep.png', dpi=150)
    plt.show()

    # -------------------------------------------------------------------------
    # 3. fNIRS-like Example (Low Frequencies)
    # -------------------------------------------------------------------------
    print("\n3. fNIRS-like Low Frequency Analysis")
    print("-" * 40)

    # Simulate fNIRS-like data (very low frequencies)
    fs_fnirs = 10  # 10 Hz sampling (typical for fNIRS)
    duration_fnirs = 120  # 2 minutes
    t3 = np.linspace(0, duration_fnirs, int(fs_fnirs * duration_fnirs), endpoint=False)

    # Neurogenic band (~0.02-0.04 Hz)
    neurogenic = 0.5 * np.sin(2 * np.pi * 0.025 * t3)

    # Respiratory (~0.2-0.3 Hz)
    respiratory = 0.3 * np.sin(2 * np.pi * 0.25 * t3)

    # Cardiac alias (~1 Hz, but often aliased in fNIRS)
    cardiac = 0.2 * np.sin(2 * np.pi * 1.0 * t3)

    # Combine with slow drift and noise
    fnirs_signal = neurogenic + respiratory + cardiac + 0.1 * np.random.randn(len(t3))

    # Add slow drift
    fnirs_signal += 0.3 * np.sin(2 * np.pi * 0.005 * t3)

    # Compute spectrogram with different cycle parameters
    spec_fnirs_low = cmt_spectrogram(fnirs_signal, fs_fnirs, freq_range=(0.01, 1.5),
                                      n_freqs=50, n_cycles=3)
    spec_fnirs_high = cmt_spectrogram(fnirs_signal, fs_fnirs, freq_range=(0.01, 1.5),
                                       n_freqs=50, n_cycles=7)

    print(f"fNIRS signal: {duration_fnirs}s at {fs_fnirs}Hz")
    print(f"Components: Neurogenic (0.025Hz), Respiratory (0.25Hz), Cardiac (1Hz)")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(t3, fnirs_signal, 'k-', linewidth=0.5)
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Simulated fNIRS Signal')
    axes[0].set_xlim([0, duration_fnirs])

    plot_spectrogram(spec_fnirs_low, title='CMT Spectrogram (n_cycles=3, better time resolution)', ax=axes[1])
    plot_spectrogram(spec_fnirs_high, title='CMT Spectrogram (n_cycles=7, better frequency resolution)', ax=axes[2])

    # Add frequency band annotations
    for ax in axes[1:]:
        ax.axhline(y=0.025, color='white', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=0.25, color='white', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=1.0, color='white', linestyle='--', alpha=0.7, linewidth=1)

    plt.tight_layout()
    plt.savefig('demo_3_fnirs_analysis.png', dpi=150)
    plt.show()

    print("\n" + "=" * 60)
    print("Demo complete! Generated figures:")
    print("  - demo_1_basic_spectrogram.png")
    print("  - demo_2_parameter_sweep.png")
    print("  - demo_3_fnirs_analysis.png")
    print("=" * 60)

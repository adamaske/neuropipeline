import numpy as np
import matplotlib.pyplot as plt
from .fnirs import fNIRS
import pywt


def plot_phase_analysis(fnirs:fNIRS, freq_bands=[(0.01, 0.2),], freq_names=["Band 1"], n_cycles=7):

    # We need to open a whole new matplotlib window

    # First lets only consider hbo data
    # First plot every channel
    hbo_data, hbo_names, hbr_data, hbr_names = fnirs.split()

    n_bands = len(freq_bands)
    n_cols = min(n_bands, 2)
    n_rows = (n_bands + 1) // 2  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)

    # Flatten axes for easy indexing
    axes_flat = axes.flatten()

    for i, (freq_band, freq_name) in enumerate(zip(freq_bands, freq_names)):
        print(f"Computing connectivity in band {freq_band} : {freq_name}")

        mat = connectivity_matrix(hbo_data, fnirs.sampling_frequency, freq_band, n_cycles)


        plot_connectivity_matrix(mat, hbo_names,
                            title=f'{freq_name} ({freq_band[0]}-{freq_band[1]} Hz)', ax=axes_flat[i])
            
    
    plt.tight_layout()
    plt.savefig('fnirs_connectivity.png', dpi=150)
    plt.show()
    
    # Now, how about the phase locking between two channels over time?
    
    pass

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

# Usage from visualizer
# plot_connectivity_matrix

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


def phase_coherence(signal1, signal2, fs, freqs, n_cycles=5):
    """
    Compute phase-locking value (PLV) between two signals across frequencies.
    
    Parameters:
    -----------
    signal1, signal2 : array-like
        Input signals (must be same length)
    fs : float
        Sampling frequency in Hz
    freqs : array-like
        Frequencies of interest in Hz
    n_cycles : float
        Number of cycles in the wavelet
    
    Returns:
    --------
    plv : ndarray
        Phase-locking values for each frequency (0 to 1)
    freqs : ndarray
        Corresponding frequencies
    """
    # Get complex CMT coefficients for both signals
    coef1 = cmt(signal1, fs, freqs, n_cycles)
    coef2 = cmt(signal2, fs, freqs, n_cycles)
    
    # Extract instantaneous phase
    phase1 = np.angle(coef1)
    phase2 = np.angle(coef2)
    
    # PLV = |mean(exp(i * phase_diff))|
    phase_diff = phase1 - phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=1))
    
    return plv, freqs


def phase_coherence_time_resolved(signal1, signal2, fs, freqs, n_cycles=5, window_size=None):
    """
    Compute time-resolved phase-locking value between two signals.
    
    Parameters:
    -----------
    signal1, signal2 : array-like
        Input signals (must be same length)
    fs : float
        Sampling frequency in Hz
    freqs : array-like
        Frequencies of interest in Hz
    n_cycles : float
        Number of cycles in the wavelet
    window_size : int or None
        Sliding window size for time-resolved PLV. If None, uses frequency-adaptive window.
    
    Returns:
    --------
    plv_matrix : ndarray
        Time-resolved PLV (n_freqs, n_times)
    freqs : ndarray
        Frequencies
    times : ndarray
        Time points
    """
    coef1 = cmt(signal1, fs, freqs, n_cycles)
    coef2 = cmt(signal2, fs, freqs, n_cycles)
    
    phase_diff = np.angle(coef1) - np.angle(coef2)
    n_freqs, n_times = phase_diff.shape
    
    plv_matrix = np.zeros_like(phase_diff)
    
    for i, freq in enumerate(freqs):
        # Adaptive window: ~3 cycles at each frequency
        if window_size is None:
            win = max(int(3 * fs / freq), 5)
        else:
            win = window_size
        
        half_win = win // 2
        
        for t in range(n_times):
            t_start = max(0, t - half_win)
            t_end = min(n_times, t + half_win + 1)
            plv_matrix[i, t] = np.abs(np.mean(np.exp(1j * phase_diff[i, t_start:t_end])))
    
    times = np.arange(n_times) / fs
    return plv_matrix, freqs, times


def connectivity_matrix(signals, fs, freq_band, n_cycles=5):
    """
    Compute all-to-all phase coherence connectivity matrix.
    
    Parameters:
    -----------
    signals : ndarray
        Input signals (n_channels, n_times)
    fs : float
        Sampling frequency in Hz
    freq_band : tuple
        (min_freq, max_freq) - PLV will be averaged over this band
    n_cycles : float
        Number of cycles in the wavelet
    
    Returns:
    --------
    conn_matrix : ndarray
        Connectivity matrix (n_channels, n_channels)
    """
    n_channels = signals.shape[0]
    freqs = np.linspace(freq_band[0], freq_band[1], 10)
    
    conn_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            plv, _ = phase_coherence(signals[i], signals[j], fs, freqs, n_cycles)
            mean_plv = np.mean(plv)
            conn_matrix[i, j] = mean_plv
            conn_matrix[j, i] = mean_plv
    
    np.fill_diagonal(conn_matrix, 1.0)
    return conn_matrix


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


def plot_phase_coherence(plv, freqs, title='Phase-Locking Value', ax=None):
    """
    Plot phase coherence spectrum.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(freqs, plv, 'b-', linewidth=2)
    ax.fill_between(freqs, 0, plv, alpha=0.3)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PLV')
    ax.set_title(title)
    ax.set_ylim([0, 1])
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='PLV = 0.5')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_connectivity_matrix(conn_matrix, channel_names=None, title='Phase Coherence Connectivity', ax=None):
    """
    Plot connectivity matrix as heatmap.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    n_channels = conn_matrix.shape[0]
    if channel_names is None:
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]

    im = ax.imshow(conn_matrix, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(n_channels))
    ax.set_yticks(range(n_channels))
    ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.set_yticklabels(channel_names)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='PLV')

    # Add text annotations
    #for i in range(n_channels):
    #    for j in range(n_channels):
    #        ax.text(j, i, f'{conn_matrix[i, j]:.2f}', ha='center', va='center',
    #               color='white' if conn_matrix[i, j] > 0.5 else 'black', fontsize=8)

    return ax


def plot_phase_locking_time(fnirs, ch1_idx, ch2_idx, freq_band, n_cycles=5,
                            window_size=None, cmap='hot', figsize=(14, 8)):
    """
    Plot time-resolved phase locking value between two channels.

    Parameters:
    -----------
    fnirs : fNIRS
        fNIRS data object
    ch1_idx : int
        Index of first channel
    ch2_idx : int
        Index of second channel
    freq_band : tuple
        (min_freq, max_freq) in Hz
    n_cycles : float
        Number of cycles in the wavelet
    window_size : int or None
        Sliding window size for time-resolved PLV. If None, uses frequency-adaptive window.
    cmap : str
        Colormap for the PLV heatmap
    figsize : tuple
        Figure size

    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    hbo_data, hbo_names, hbr_data, hbr_names = fnirs.split()
    fs = fnirs.sampling_frequency

    # Get channel names
    ch1_name = hbo_names[ch1_idx] if ch1_idx < len(hbo_names) else f"Ch{ch1_idx}"
    ch2_name = hbo_names[ch2_idx] if ch2_idx < len(hbo_names) else f"Ch{ch2_idx}"

    # Extract signals
    signal1 = hbo_data[ch1_idx]
    signal2 = hbo_data[ch2_idx]

    # Define frequencies within band
    n_freqs = max(20, int((freq_band[1] - freq_band[0]) * 100))
    freqs = np.linspace(freq_band[0], freq_band[1], n_freqs)
    freqs = freqs[freqs > 0]  # Avoid zero frequency

    # Compute time-resolved PLV
    plv_matrix, freqs_out, times = phase_coherence_time_resolved(
        signal1, signal2, fs, freqs, n_cycles, window_size
    )

    # Also compute band-averaged PLV over time
    plv_band_avg = np.mean(plv_matrix, axis=0)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize,
                              gridspec_kw={'height_ratios': [1, 2, 1]})

    # Plot 1: Original signals
    ax1 = axes[0]
    t = np.arange(len(signal1)) / fs
    ax1.plot(t, signal1, 'b-', linewidth=0.8, alpha=0.7, label=ch1_name)
    ax1.plot(t, signal2, 'r-', linewidth=0.8, alpha=0.7, label=ch2_name)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('HbO')
    ax1.set_title(f'Signals: {ch1_name} vs {ch2_name}')
    ax1.legend(loc='upper right')
    ax1.set_xlim([times[0], times[-1]])

    # Plot 2: Time-frequency PLV
    ax2 = axes[1]
    im = ax2.pcolormesh(times, freqs_out, plv_matrix, shading='auto', cmap=cmap, vmin=0, vmax=1)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_title(f'Time-Resolved Phase-Locking Value ({freq_band[0]}-{freq_band[1]} Hz)')
    plt.colorbar(im, ax=ax2, label='PLV')

    # Plot 3: Band-averaged PLV over time
    ax3 = axes[2]
    ax3.plot(times, plv_band_avg, 'k-', linewidth=1.5)
    ax3.fill_between(times, 0, plv_band_avg, alpha=0.3)
    ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='PLV = 0.5')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('PLV')
    ax3.set_title(f'Band-Averaged PLV ({freq_band[0]}-{freq_band[1]} Hz)')
    ax3.set_ylim([0, 1])
    ax3.set_xlim([times[0], times[-1]])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_phase_locking_time_multi(fnirs, channel_pairs, freq_band, n_cycles=5,
                                   window_size=None, cmap='hot', figsize=(14, 10)):
    """
    Plot time-resolved phase locking for multiple channel pairs.

    Parameters:
    -----------
    fnirs : fNIRS
        fNIRS data object
    channel_pairs : list of tuples
        List of (ch1_idx, ch2_idx) pairs
    freq_band : tuple
        (min_freq, max_freq) in Hz
    n_cycles : float
        Number of cycles in the wavelet
    window_size : int or None
        Sliding window size for time-resolved PLV
    cmap : str
        Colormap for the PLV heatmap
    figsize : tuple
        Figure size

    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    hbo_data, hbo_names, hbr_data, hbr_names = fnirs.split()
    fs = fnirs.sampling_frequency

    n_pairs = len(channel_pairs)
    n_freqs = max(20, int((freq_band[1] - freq_band[0]) * 100))
    freqs = np.linspace(freq_band[0], freq_band[1], n_freqs)
    freqs = freqs[freqs > 0]

    fig, axes = plt.subplots(n_pairs, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, (ch1_idx, ch2_idx) in enumerate(channel_pairs):
        ch1_name = hbo_names[ch1_idx] if ch1_idx < len(hbo_names) else f"Ch{ch1_idx}"
        ch2_name = hbo_names[ch2_idx] if ch2_idx < len(hbo_names) else f"Ch{ch2_idx}"

        signal1 = hbo_data[ch1_idx]
        signal2 = hbo_data[ch2_idx]

        plv_matrix, freqs_out, times = phase_coherence_time_resolved(
            signal1, signal2, fs, freqs, n_cycles, window_size
        )

        ax = axes[idx]
        im = ax.pcolormesh(times, freqs_out, plv_matrix, shading='auto', cmap=cmap, vmin=0, vmax=1)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title(f'PLV: {ch1_name} ↔ {ch2_name}')
        plt.colorbar(im, ax=ax, label='PLV')

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
    # 3. Phase Coherence Analysis
    # -------------------------------------------------------------------------
    print("\n3. Phase Coherence Analysis")
    print("-" * 40)
    
    # Create two signals with known phase relationship
    t3 = np.linspace(0, 10, int(fs * 10), endpoint=False)
    
    # Signal 1: 10 Hz oscillation
    sig1 = np.sin(2 * np.pi * 10 * t3)
    
    # Signal 2: Same 10 Hz but phase-locked, plus independent 25 Hz
    sig2 = np.sin(2 * np.pi * 10 * t3 + np.pi/4) + 0.5 * np.sin(2 * np.pi * 25 * t3 + np.random.rand() * 2 * np.pi)
    
    # Add noise
    sig1 += 0.3 * np.random.randn(len(sig1))
    sig2 += 0.3 * np.random.randn(len(sig2))
    
    # Compute phase coherence
    freqs = np.linspace(1, 40, 40)
    plv, _ = phase_coherence(sig1, sig2, fs, freqs, n_cycles=7)
    
    print(f"PLV at 10 Hz: {plv[np.argmin(np.abs(freqs - 10))]:.3f} (expect high - phase-locked)")
    print(f"PLV at 25 Hz: {plv[np.argmin(np.abs(freqs - 25))]:.3f} (expect low - independent)")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    axes[0].plot(t3[:int(fs*2)], sig1[:int(fs*2)], 'b-', label='Signal 1', alpha=0.7)
    axes[0].plot(t3[:int(fs*2)], sig2[:int(fs*2)], 'r-', label='Signal 2', alpha=0.7)
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Two Signals (first 2 seconds)')
    axes[0].legend()
    
    plot_phase_coherence(plv, freqs, title='Phase-Locking Value Between Signals', ax=axes[1])
    axes[1].axvline(x=10, color='g', linestyle='--', alpha=0.7, label='10 Hz (phase-locked)')
    axes[1].axvline(x=25, color='orange', linestyle='--', alpha=0.7, label='25 Hz (independent)')
    axes[1].legend()
    
    # Time-resolved PLV
    plv_time, _, times_plv = phase_coherence_time_resolved(sig1, sig2, fs, freqs, n_cycles=7)
    im = axes[2].pcolormesh(times_plv, freqs, plv_time, shading='auto', cmap='hot', vmin=0, vmax=1)
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Frequency [Hz]')
    axes[2].set_title('Time-Resolved Phase-Locking Value')
    plt.colorbar(im, ax=axes[2], label='PLV')
    
    plt.tight_layout()
    plt.savefig('demo_3_phase_coherence.png', dpi=150)
    plt.show()
    
    # -------------------------------------------------------------------------
    # 4. Connectivity Matrix
    # -------------------------------------------------------------------------
    print("\n4. Connectivity Matrix")
    print("-" * 40)
    
    # Simulate 6 channels with known connectivity structure
    n_channels = 6
    n_samples = int(fs * 10)
    t4 = np.linspace(0, 10, n_samples, endpoint=False)
    
    # Create base oscillations
    base_10hz = np.sin(2 * np.pi * 10 * t4)
    base_25hz = np.sin(2 * np.pi * 25 * t4)
    
    signals = np.zeros((n_channels, n_samples))
    
    # Channels 0, 1, 2 share 10 Hz component (high coherence)
    signals[0] = base_10hz + 0.3 * np.random.randn(n_samples)
    signals[1] = base_10hz + np.pi/6 + 0.3 * np.random.randn(n_samples)  # slight phase shift
    signals[2] = base_10hz + np.pi/4 + 0.3 * np.random.randn(n_samples)
    
    # Channels 3, 4, 5 share 25 Hz component (different cluster)
    signals[3] = base_25hz + 0.3 * np.random.randn(n_samples)
    signals[4] = base_25hz + np.pi/6 + 0.3 * np.random.randn(n_samples)
    signals[5] = base_25hz + np.pi/4 + 0.3 * np.random.randn(n_samples)
    
    # Compute connectivity in 10 Hz band
    conn_10hz = connectivity_matrix(signals, fs, freq_band=(8, 12), n_cycles=7)
    
    # Compute connectivity in 25 Hz band
    conn_25hz = connectivity_matrix(signals, fs, freq_band=(23, 27), n_cycles=7)
    
    channel_names = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6']
    
    print("Expected: High coherence within Ch1-3 at 10Hz, within Ch4-6 at 25Hz")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_connectivity_matrix(conn_10hz, channel_names, 
                            title='Connectivity Matrix (8-12 Hz band)', ax=axes[0])
    plot_connectivity_matrix(conn_25hz, channel_names,
                            title='Connectivity Matrix (23-27 Hz band)', ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('demo_4_connectivity.png', dpi=150)
    plt.show()
    
    # -------------------------------------------------------------------------
    # 5. fNIRS-like Example (Low Frequencies)
    # -------------------------------------------------------------------------
    print("\n5. fNIRS-like Low Frequency Analysis")
    print("-" * 40)
    
    # Simulate fNIRS-like data (very low frequencies)
    fs_fnirs = 10  # 10 Hz sampling (typical for fNIRS)
    duration_fnirs = 120  # 2 minutes
    t5 = np.linspace(0, duration_fnirs, int(fs_fnirs * duration_fnirs), endpoint=False)
    
    # Neurogenic band (~0.02-0.04 Hz)
    neurogenic = 0.5 * np.sin(2 * np.pi * 0.025 * t5)
    
    # Respiratory (~0.2-0.3 Hz)
    respiratory = 0.3 * np.sin(2 * np.pi * 0.25 * t5)
    
    # Cardiac alias (~1 Hz, but often aliased in fNIRS)
    cardiac = 0.2 * np.sin(2 * np.pi * 1.0 * t5)
    
    # Combine with slow drift and noise
    fnirs_signal = neurogenic + respiratory + cardiac + 0.1 * np.random.randn(len(t5))
    
    # Add slow drift
    fnirs_signal += 0.3 * np.sin(2 * np.pi * 0.005 * t5)
    
    # Compute spectrogram with different cycle parameters
    spec_fnirs_low = cmt_spectrogram(fnirs_signal, fs_fnirs, freq_range=(0.01, 1.5), 
                                      n_freqs=50, n_cycles=3)
    spec_fnirs_high = cmt_spectrogram(fnirs_signal, fs_fnirs, freq_range=(0.01, 1.5), 
                                       n_freqs=50, n_cycles=7)
    
    print(f"fNIRS signal: {duration_fnirs}s at {fs_fnirs}Hz")
    print(f"Components: Neurogenic (0.025Hz), Respiratory (0.25Hz), Cardiac (1Hz)")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    axes[0].plot(t5, fnirs_signal, 'k-', linewidth=0.5)
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
    plt.savefig('demo_5_fnirs_analysis.png', dpi=150)
    plt.show()
    
    print("\n" + "=" * 60)
    print("Demo complete! Generated figures:")
    print("  - demo_1_basic_spectrogram.png")
    print("  - demo_2_parameter_sweep.png")
    print("  - demo_3_phase_coherence.png")
    print("  - demo_4_connectivity.png")
    print("  - demo_5_fnirs_analysis.png")
    print("=" * 60)
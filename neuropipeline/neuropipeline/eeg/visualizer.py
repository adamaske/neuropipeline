"""
Interactive GUI visualizer for EEG data.

Usage:
    from neuropipeline.eeg import visualizer
    visualizer.open(eeg_path_or_object)
"""
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.signal import spectrogram
import pywt
from typing import Union

from .eeg import EEG

# Marker color palette for distinguishing different marker types
MARKER_COLORS = [
    '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000'
]

# Module-level configuration
_config = {
    'spectrogram_freq_min': 0.5,
    'spectrogram_freq_max': 50.0,
    'spectrum_freq_max': 100.0,
    'marker_dictionary': {},
    'spectrum_mode': 'FFT',
    'spectrogram_method': 'STFT',
}


def compute_fft(time_series, fs, freq_limit: float | None):
    """Compute FFT of a time series."""
    N = len(time_series)
    fft_result = np.fft.fft(time_series)
    fft_freq = np.fft.fftfreq(N, d=1/fs)

    # Take the positive half of the spectrum
    positive_freqs = fft_freq[:N // 2]
    positive_spectrum = np.abs(fft_result[:N // 2]) * (2 / N)

    if freq_limit is None:
        return positive_freqs, positive_spectrum

    # Limit to specified frequency range
    freq_mask = positive_freqs <= freq_limit
    return positive_freqs[freq_mask], positive_spectrum[freq_mask]


def compute_psd(time_series, fs, freq_limit: float | None):
    """Compute Power Spectral Density of a time series."""
    freqs, spectrum = compute_fft(time_series, fs, freq_limit)
    psd = np.square(spectrum) / (fs * len(time_series))
    psd[1:] = 2 * psd[1:]
    return freqs, psd


class EEGVisualizer:
    """Interactive GUI for visualizing EEG channel data."""

    def __init__(self, data: Union[str, EEG]):
        """
        Initialize the visualizer.

        Args:
            data: Either a path to an HDF5 file or an EEG object
        """
        if isinstance(data, str):
            self.eeg = EEG(data)
        elif isinstance(data, EEG):
            self.eeg = data
        else:
            raise TypeError("data must be an HDF5 file path (str) or EEG object")

        self.current_channel = 0
        self.fs = self.eeg.sampling_frequency
        self.num_channels = self.eeg.channel_num

        # Spectrum display mode: "PSD" or "FFT" (read from module config)
        self.spectrum_mode = _config['spectrum_mode']

        # Spectrogram method: "STFT" or "Wavelet" (read from module config)
        self.spectrogram_method = _config['spectrogram_method']

        # Frequency limits (read from module config)
        self.spectrogram_freq_min = _config['spectrogram_freq_min']
        self.spectrogram_freq_max = _config['spectrogram_freq_max']
        self.spectrum_freq_max = _config['spectrum_freq_max']

        # Marker visibility and color mapping
        self._setup_markers()

        self._setup_gui()

    def _setup_markers(self):
        """Setup marker types, colors, visibility state, and labels."""
        self.marker_types = []
        self.marker_colors = {}
        self.marker_visibility = {}
        self.marker_labels = {}
        self.marker_checkboxes = {}
        self.marker_vars = {}

        # Get marker dictionary from config
        marker_dict = _config.get('marker_dictionary', {})

        if self.eeg.feature_descriptions is not None and len(self.eeg.feature_descriptions) > 0:
            # Get unique marker types
            self.marker_types = sorted(set(self.eeg.feature_descriptions))

            # Assign colors, labels, and initialize visibility
            for i, marker_type in enumerate(self.marker_types):
                color_idx = i % len(MARKER_COLORS)
                self.marker_colors[marker_type] = MARKER_COLORS[color_idx]
                self.marker_visibility[marker_type] = True
                self.marker_labels[marker_type] = marker_dict.get(marker_type, f"Marker {marker_type}")

    def _setup_gui(self):
        """Setup the main GUI window."""
        self.root = tk.Tk()
        self.root.title("EEG Visualizer - Neuropipeline")
        self.root.geometry("1200x900")
        self.root.configure(bg='#2b2b2b')

        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white', font=('Segoe UI', 12))
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'))
        style.configure('TButton', font=('Segoe UI', 14))

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header with channel navigation
        self._create_header(main_frame)

        # Plot area
        self._create_plot_area(main_frame)

        # Initial plot
        self._update_plots()

        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self._prev_channel())
        self.root.bind('<Right>', lambda e: self._next_channel())
        self.root.bind('<Escape>', lambda e: self.root.destroy())
        self.root.bind('<space>', lambda e: self._toggle_spectrum_mode())

    def _create_header(self, parent):
        """Create the header with channel navigation."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        # Left arrow button
        self.prev_btn = tk.Button(
            header_frame,
            text="◀",
            font=('Segoe UI', 20),
            command=self._prev_channel,
            bg='#404040',
            fg='white',
            activebackground='#505050',
            activeforeground='white',
            relief=tk.FLAT,
            width=3,
            cursor='hand2'
        )
        self.prev_btn.pack(side=tk.LEFT, padx=10)

        # Channel label (centered)
        self.channel_label = ttk.Label(
            header_frame,
            text=self._get_channel_text(),
            style='Title.TLabel',
            anchor='center'
        )
        self.channel_label.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Right arrow button
        self.next_btn = tk.Button(
            header_frame,
            text="▶",
            font=('Segoe UI', 20),
            command=self._next_channel,
            bg='#404040',
            fg='white',
            activebackground='#505050',
            activeforeground='white',
            relief=tk.FLAT,
            width=3,
            cursor='hand2'
        )
        self.next_btn.pack(side=tk.RIGHT, padx=10)

        # Channel slider frame
        slider_frame = ttk.Frame(parent)
        slider_frame.pack(fill=tk.X, pady=(0, 5))

        # Channel slider
        self.channel_var = tk.IntVar(value=1)
        self.channel_slider = tk.Scale(
            slider_frame,
            from_=1,
            to=self.num_channels,
            orient=tk.HORIZONTAL,
            variable=self.channel_var,
            command=self._on_slider_change,
            bg='#2b2b2b',
            fg='white',
            highlightthickness=0,
            troughcolor='#404040',
            activebackground='#505050',
            font=('Segoe UI', 10),
            showvalue=False
        )
        self.channel_slider.pack(fill=tk.X, padx=20, expand=True)

        # Info label
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.X, pady=(0, 5))

        info_text = f"Sampling Rate: {self.fs:.2f} Hz  |  Channels: {self.num_channels}  |  Duration: {self._get_duration():.2f} s"
        self.info_label = ttk.Label(info_frame, text=info_text, anchor='center')
        self.info_label.pack(expand=True)

        # Controls frame
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, pady=(0, 5))

        # Spectrum mode dropdown
        spectrum_label = ttk.Label(controls_frame, text="Spectrum:", font=('Segoe UI', 11))
        spectrum_label.pack(side=tk.LEFT, padx=(20, 5))

        self.spectrum_mode_var = tk.StringVar(value=self.spectrum_mode)
        self.spectrum_dropdown = ttk.Combobox(
            controls_frame,
            textvariable=self.spectrum_mode_var,
            values=["FFT", "PSD"],
            state="readonly",
            width=8,
            font=('Segoe UI', 10)
        )
        self.spectrum_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        self.spectrum_dropdown.bind("<<ComboboxSelected>>", self._on_spectrum_dropdown_change)

        # Spectrogram method dropdown
        spectrogram_label = ttk.Label(controls_frame, text="Spectrogram:", font=('Segoe UI', 11))
        spectrogram_label.pack(side=tk.LEFT, padx=(10, 5))

        self.spectrogram_method_var = tk.StringVar(value=self.spectrogram_method)
        self.spectrogram_dropdown = ttk.Combobox(
            controls_frame,
            textvariable=self.spectrogram_method_var,
            values=["STFT", "Wavelet"],
            state="readonly",
            width=8,
            font=('Segoe UI', 10)
        )
        self.spectrogram_dropdown.pack(side=tk.LEFT, padx=(0, 20))
        self.spectrogram_dropdown.bind("<<ComboboxSelected>>", self._on_spectrogram_dropdown_change)

        # Marker toggle frame (only if markers exist)
        if len(self.marker_types) > 0:
            marker_frame = ttk.Frame(parent)
            marker_frame.pack(fill=tk.X, pady=(0, 5))

            marker_label = ttk.Label(marker_frame, text="Markers:", font=('Segoe UI', 11))
            marker_label.pack(side=tk.LEFT, padx=(20, 10))

            for marker_type in self.marker_types:
                color = self.marker_colors[marker_type]
                label = self.marker_labels[marker_type]
                var = tk.BooleanVar(value=True)
                self.marker_vars[marker_type] = var

                cb = tk.Checkbutton(
                    marker_frame,
                    text=label,
                    variable=var,
                    command=lambda mt=marker_type: self._on_marker_toggle(mt),
                    bg='#2b2b2b',
                    fg=color,
                    selectcolor='#404040',
                    activebackground='#2b2b2b',
                    activeforeground=color,
                    font=('Segoe UI', 10, 'bold'),
                    cursor='hand2'
                )
                cb.pack(side=tk.LEFT, padx=5)
                self.marker_checkboxes[marker_type] = cb

    def _create_plot_area(self, parent):
        """Create the matplotlib plot area."""
        plot_container = ttk.Frame(parent)
        plot_container.pack(fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 10), facecolor='#2b2b2b')
        self.plot_container = plot_container

        self._setup_axes()

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_container)
        self.canvas.draw()

        # Add navigation toolbar
        toolbar_frame = ttk.Frame(plot_container)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _setup_axes(self):
        """Setup axes for the plots."""
        self.fig.clear()

        # 3 rows: timeseries, spectrogram, spectrum
        gs = self.fig.add_gridspec(3, 1, hspace=0.4, left=0.08, right=0.95, top=0.95, bottom=0.08)
        self.ax_timeseries = self.fig.add_subplot(gs[0])
        self.ax_spectrogram = self.fig.add_subplot(gs[1])
        self.ax_spectrum = self.fig.add_subplot(gs[2])

        axes_list = [self.ax_timeseries, self.ax_spectrogram, self.ax_spectrum]

        # Style all axes
        for ax in axes_list:
            self._style_axis(ax)

    def _style_axis(self, ax):
        """Apply dark theme styling to an axis."""
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#555555')

    def _get_channel_text(self) -> str:
        """Get the current channel display text."""
        return f"Channel {self.current_channel + 1}/{self.num_channels}"

    def _get_duration(self) -> float:
        """Get the total duration of the recording in seconds."""
        return self.eeg.channel_data.shape[1] / self.fs

    def _get_marker_time(self, onset_sample: int) -> float:
        """Convert marker onset from samples to seconds."""
        return onset_sample / self.fs

    def _on_slider_change(self, value):
        """Handle slider value change."""
        new_channel = int(value) - 1
        if new_channel != self.current_channel:
            self.current_channel = new_channel
            self._update_plots()

    def _prev_channel(self):
        """Navigate to the previous channel."""
        if self.current_channel > 0:
            self.current_channel -= 1
            self.channel_slider.set(self.current_channel + 1)
            self._update_plots()

    def _next_channel(self):
        """Navigate to the next channel."""
        if self.current_channel < self.num_channels - 1:
            self.current_channel += 1
            self.channel_slider.set(self.current_channel + 1)
            self._update_plots()

    def _on_marker_toggle(self, marker_type):
        """Handle marker visibility toggle."""
        self.marker_visibility[marker_type] = self.marker_vars[marker_type].get()
        self._update_plots()

    def _on_spectrum_dropdown_change(self, event=None):
        """Handle spectrum mode dropdown change."""
        self.spectrum_mode = self.spectrum_mode_var.get()
        self._update_spectrum_only()

    def _on_spectrogram_dropdown_change(self, event=None):
        """Handle spectrogram method dropdown change."""
        self.spectrogram_method = self.spectrogram_method_var.get()
        self._update_spectrogram_only()

    def _toggle_spectrum_mode(self):
        """Toggle between PSD and FFT spectrum display (keyboard shortcut)."""
        self.spectrum_mode = "FFT" if self.spectrum_mode == "PSD" else "PSD"
        self.spectrum_mode_var.set(self.spectrum_mode)
        self._update_spectrum_only()

    def _update_spectrogram_only(self):
        """Update only the spectrogram plot."""
        data = self.eeg.channel_data[self.current_channel]
        self.ax_spectrogram.clear()
        self._plot_spectrogram(data)
        self._style_axis(self.ax_spectrogram)
        self.canvas.draw()

    def _update_spectrum_only(self):
        """Update only the spectrum plot."""
        data = self.eeg.channel_data[self.current_channel]
        self.ax_spectrum.clear()
        self._plot_spectrum(data)
        self._style_axis(self.ax_spectrum)
        self.canvas.draw()

    def _update_plots(self):
        """Update all plots for the current channel."""
        # Update channel label
        self.channel_label.config(text=self._get_channel_text())

        # Get current channel data
        data = self.eeg.channel_data[self.current_channel]
        time = np.arange(len(data)) / self.fs

        # Clear all axes
        self.ax_timeseries.clear()
        self.ax_spectrogram.clear()
        self.ax_spectrum.clear()

        # Plot timeseries
        self._plot_timeseries(time, data)

        # Plot spectrogram
        self._plot_spectrogram(data)

        # Plot frequency spectrum
        self._plot_spectrum(data)

        # Re-apply styling
        for ax in [self.ax_timeseries, self.ax_spectrogram, self.ax_spectrum]:
            self._style_axis(ax)

        self.canvas.draw()

    def _plot_timeseries(self, time: np.ndarray, data: np.ndarray):
        """Plot the timeseries with event markers."""
        self.ax_timeseries.plot(time, data, color='#4fc3f7', linewidth=0.5)
        self.ax_timeseries.set_xlabel('Time (s)')
        self.ax_timeseries.set_ylabel(f'Amplitude ({self.eeg.acquisition_unit or "μV"})')
        self.ax_timeseries.set_title('Time Series')
        self.ax_timeseries.grid(True, alpha=0.3, color='#555555')

        # Add feature/event markers if available
        marker_handles = []
        marker_legend_labels = []
        if self.eeg.feature_onsets is not None and len(self.eeg.feature_onsets) > 0:
            added_to_legend = set()

            for i, onset in enumerate(self.eeg.feature_onsets):
                onset_time = self._get_marker_time(onset)
                if 0 <= onset_time <= time[-1]:
                    marker_type = self.eeg.feature_descriptions[i]

                    if not self.marker_visibility.get(marker_type, True):
                        continue

                    color = self.marker_colors.get(marker_type, '#ffcc00')
                    line = self.ax_timeseries.axvline(
                        x=onset_time, color=color, linestyle='--', alpha=0.7, linewidth=1
                    )

                    if marker_type not in added_to_legend:
                        marker_handles.append(line)
                        label = self.marker_labels.get(marker_type, f"Marker {marker_type}")
                        marker_legend_labels.append(label)
                        added_to_legend.add(marker_type)

        if marker_handles:
            self.ax_timeseries.legend(
                marker_handles, marker_legend_labels, loc='upper right',
                facecolor='#2b2b2b', edgecolor='#555555', labelcolor='white'
            )

        self.ax_timeseries.set_xlim(0, time[-1])

    def _plot_spectrogram(self, data: np.ndarray):
        """Plot the spectrogram of the signal."""
        freq_min = self.spectrogram_freq_min
        freq_max = min(self.spectrogram_freq_max, self.fs / 2)

        if self.spectrogram_method == "STFT":
            self._plot_stft_spectrogram(data, freq_min, freq_max)
        else:
            self._plot_wavelet_spectrogram(data, freq_min, freq_max)

    def _plot_stft_spectrogram(self, data: np.ndarray, freq_min: float, freq_max: float):
        """Plot STFT-based spectrogram."""
        # Use appropriate window size for EEG (typically higher sampling rate)
        nperseg = min(int(self.fs * 2), len(data) // 4)
        if nperseg < 4:
            nperseg = len(data)

        f, t, Sxx = spectrogram(data, fs=self.fs, nperseg=nperseg, noverlap=nperseg // 2)
        freq_mask = (f >= freq_min) & (f <= freq_max)

        Sxx_plot = Sxx[freq_mask, :]
        if Sxx_plot.max() > 0:
            Sxx_db = 10 * np.log10(Sxx_plot + 1e-12)
        else:
            Sxx_db = Sxx_plot

        self.ax_spectrogram.pcolormesh(t, f[freq_mask], Sxx_db, shading='gouraud', cmap='viridis')
        self.ax_spectrogram.set_xlabel('Time (s)')
        self.ax_spectrogram.set_ylabel('Frequency (Hz)')
        self.ax_spectrogram.set_title('Spectrogram (STFT)')

    def _plot_wavelet_spectrogram(self, data: np.ndarray, freq_min: float, freq_max: float):
        """Plot wavelet-based spectrogram using continuous wavelet transform."""
        wavelet_name = 'cmor1.5-1.0'
        num_freqs = 64
        frequencies = np.linspace(max(freq_min, 0.5), freq_max, num_freqs)

        sampling_period = 1.0 / self.fs
        scales = pywt.frequency2scale(wavelet_name, frequencies * sampling_period)

        coef, freqs_out = pywt.cwt(data, scales, wavelet_name, sampling_period=sampling_period)
        power = np.abs(coef) ** 2

        t = np.arange(len(data)) / self.fs

        if power.max() > 0:
            power_db = 10 * np.log10(power + 1e-12)
        else:
            power_db = power

        self.ax_spectrogram.pcolormesh(t, freqs_out, power_db, shading='gouraud', cmap='viridis')
        self.ax_spectrogram.set_xlabel('Time (s)')
        self.ax_spectrogram.set_ylabel('Frequency (Hz)')
        self.ax_spectrogram.set_title('Spectrogram (Wavelet)')

    def _plot_spectrum(self, data: np.ndarray):
        """Plot the frequency spectrum (PSD or FFT based on current mode)."""
        freq_limit = min(self.spectrum_freq_max, self.fs / 2)

        if self.spectrum_mode == "PSD":
            freqs, spectrum = compute_psd(data, self.fs, freq_limit)
            self.ax_spectrum.semilogy(freqs, spectrum, color='#4fc3f7', linewidth=1.2)
            self.ax_spectrum.set_ylabel('Power Spectral Density')
            self.ax_spectrum.set_title('Frequency Spectrum (PSD)')
        else:
            freqs, spectrum = compute_fft(data, self.fs, freq_limit)
            self.ax_spectrum.plot(freqs, spectrum, color='#4fc3f7', linewidth=1.2)
            self.ax_spectrum.set_ylabel('Amplitude')
            self.ax_spectrum.set_title('Frequency Spectrum (FFT)')

        self.ax_spectrum.set_xlabel('Frequency (Hz)')
        self.ax_spectrum.grid(True, alpha=0.3, color='#555555')
        self.ax_spectrum.set_xlim(0, freq_limit)

    def set_spectrogram_limits(self, freq_min: float = 0.5, freq_max: float = 50.0):
        """
        Set the frequency limits for the spectrogram display.

        Args:
            freq_min: Minimum frequency to display (Hz). Default is 0.5.
            freq_max: Maximum frequency to display (Hz). Default is 50.0.
        """
        self.spectrogram_freq_min = freq_min
        self.spectrogram_freq_max = freq_max
        self._update_plots()

    def run(self):
        """Start the GUI event loop."""
        self.root.mainloop()


def set_spectrogram_limits(freq_min: float = 0.5, freq_max: float = 50.0):
    """
    Configure the spectrogram frequency limits for the visualizer.

    Call this before open() to set the frequency range displayed in spectrograms.

    Args:
        freq_min: Minimum frequency to display (Hz). Default is 0.5.
        freq_max: Maximum frequency to display (Hz). Default is 50.0.

    Example:
        >>> import neuropipeline.eeg_visualizer as eegv
        >>> eegv.set_spectrogram_limits(1, 30)
        >>> eegv.open(eeg)
    """
    _config['spectrogram_freq_min'] = freq_min
    _config['spectrogram_freq_max'] = freq_max


def set_spectrum_freq_max(freq_max: float = 100.0):
    """
    Configure the maximum frequency for the spectrum plot.

    Call this before open() to set the frequency range displayed in the spectrum.

    Args:
        freq_max: Maximum frequency to display (Hz). Default is 100.0.

    Example:
        >>> import neuropipeline.eeg_visualizer as eegv
        >>> eegv.set_spectrum_freq_max(60)
        >>> eegv.open(eeg)
    """
    _config['spectrum_freq_max'] = freq_max


def set_marker_dictionary(marker_dict: dict):
    """
    Configure custom labels for marker types.

    Call this before open() to set human-readable names for marker indices.
    Marker types not in the dictionary will display as "Marker {index}".

    Args:
        marker_dict: Dictionary mapping marker index (int) to label (str).

    Example:
        >>> from neuropipeline.eeg import visualizer
        >>> visualizer.set_marker_dictionary({1: "Rest", 2: "Left Hand", 3: "Right Hand"})
        >>> visualizer.open(eeg)
    """
    _config['marker_dictionary'] = marker_dict


def set_spectrum_mode(mode: str = "FFT"):
    """
    Configure the default spectrum display mode.

    Call this before open() to set whether to display FFT or PSD.

    Args:
        mode: Either "FFT" or "PSD". Default is "FFT".

    Example:
        >>> from neuropipeline.eeg import visualizer
        >>> visualizer.set_spectrum_mode("PSD")
        >>> visualizer.open(eeg)
    """
    if mode not in ("FFT", "PSD"):
        raise ValueError("mode must be 'FFT' or 'PSD'")
    _config['spectrum_mode'] = mode


def set_spectrogram_method(method: str = "STFT"):
    """
    Configure the default spectrogram computation method.

    Call this before open() to set whether to use STFT or Wavelet.
    Note: Wavelet is more computationally expensive but provides better
    time-frequency resolution. Use STFT for faster visualization.

    Args:
        method: Either "STFT" or "Wavelet". Default is "STFT".

    Example:
        >>> from neuropipeline.eeg import visualizer
        >>> visualizer.set_spectrogram_method("Wavelet")  # Higher quality but slower
        >>> visualizer.open(eeg)
    """
    if method not in ("STFT", "Wavelet"):
        raise ValueError("method must be 'STFT' or 'Wavelet'")
    _config['spectrogram_method'] = method


def open(data: Union[str, EEG]):
    """
    Open the interactive visualizer for EEG data.

    Args:
        data: Either a path to an HDF5 file (str) or an EEG object

    Example:
        >>> from neuropipeline.eeg import visualizer
        >>> visualizer.open("path/to/file.hdf5")

        >>> # With custom settings
        >>> visualizer.set_spectrogram_method("STFT")  # Faster (default for EEG)
        >>> visualizer.set_spectrum_mode("PSD")
        >>> visualizer.set_spectrogram_limits(1, 40)
        >>> visualizer.open("path/to/file.hdf5")
    """
    viz = EEGVisualizer(data)
    viz.run()

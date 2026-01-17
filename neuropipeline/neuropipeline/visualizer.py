"""
Interactive GUI visualizer for fNIRS (SNIRF) data.

Usage:
    import neuropipeline.visualizer as visualizer
    visualizer.open(snirf_path_or_object)
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.signal import spectrogram
from typing import Union

from .fnirs import fNIRS, compute_psd, compute_fft


class SNIRFVisualizer:
    """Interactive GUI for visualizing fNIRS channel data."""

    def __init__(self, data: Union[str, fNIRS]):
        """
        Initialize the visualizer.

        Args:
            data: Either a path to a SNIRF file or an fNIRS object
        """
        if isinstance(data, str):
            self.fnirs = fNIRS(data)
        elif isinstance(data, fNIRS):
            self.fnirs = data
        else:
            raise TypeError("data must be a SNIRF file path (str) or fNIRS object")

        self.current_channel = 0
        self.num_channels = self.fnirs.channel_num
        self.fs = self.fnirs.sampling_frequency

        # Spectrum display mode: "PSD" or "FFT"
        self.spectrum_mode = "PSD"

        self._setup_gui()

    def _setup_gui(self):
        """Setup the main GUI window."""
        self.root = tk.Tk()
        self.root.title("SNIRF Visualizer - Neuropipeline")
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

        # Spectrum mode toggle
        spectrum_frame = ttk.Frame(parent)
        spectrum_frame.pack(fill=tk.X, pady=(0, 5))

        # Left arrow for spectrum mode
        self.spectrum_prev_btn = tk.Button(
            spectrum_frame,
            text="◀",
            font=('Segoe UI', 12),
            command=self._toggle_spectrum_mode,
            bg='#404040',
            fg='white',
            activebackground='#505050',
            activeforeground='white',
            relief=tk.FLAT,
            width=2,
            cursor='hand2'
        )
        self.spectrum_prev_btn.pack(side=tk.LEFT, padx=(10, 5))

        # Spectrum mode label
        self.spectrum_mode_label = ttk.Label(
            spectrum_frame,
            text=f"Spectrum: {self.spectrum_mode}",
            anchor='center'
        )
        self.spectrum_mode_label.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Right arrow for spectrum mode
        self.spectrum_next_btn = tk.Button(
            spectrum_frame,
            text="▶",
            font=('Segoe UI', 12),
            command=self._toggle_spectrum_mode,
            bg='#404040',
            fg='white',
            activebackground='#505050',
            activeforeground='white',
            relief=tk.FLAT,
            width=2,
            cursor='hand2'
        )
        self.spectrum_next_btn.pack(side=tk.RIGHT, padx=(5, 10))

    def _create_plot_area(self, parent):
        """Create the matplotlib plot area with scrollable canvas."""
        # Create a canvas with scrollbar for the plots
        plot_container = ttk.Frame(parent)
        plot_container.pack(fill=tk.BOTH, expand=True)

        # Create matplotlib figure with subplots
        self.fig = Figure(figsize=(12, 10), facecolor='#2b2b2b')
        self.fig.subplots_adjust(hspace=0.4, left=0.08, right=0.95, top=0.95, bottom=0.08)

        # Create subplots
        self.ax_timeseries = self.fig.add_subplot(3, 1, 1)
        self.ax_spectrogram = self.fig.add_subplot(3, 1, 2)
        self.ax_spectrum = self.fig.add_subplot(3, 1, 3)

        # Style axes
        for ax in [self.ax_timeseries, self.ax_spectrogram, self.ax_spectrum]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#555555')

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_container)
        self.canvas.draw()

        # Add navigation toolbar
        toolbar_frame = ttk.Frame(plot_container)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _get_channel_text(self) -> str:
        """Get the current channel display text."""
        channel_name = self.fnirs.channel_names[self.current_channel]
        return f"Channel {self.current_channel + 1}/{self.num_channels}: {channel_name}"

    def _get_duration(self) -> float:
        """Get the total duration of the recording in seconds."""
        return self.fnirs.channel_data.shape[1] / self.fs

    def _on_slider_change(self, value):
        """Handle slider value change."""
        new_channel = int(value) - 1  # Slider is 1-indexed, internal is 0-indexed
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

    def _toggle_spectrum_mode(self):
        """Toggle between PSD and FFT spectrum display."""
        self.spectrum_mode = "FFT" if self.spectrum_mode == "PSD" else "PSD"
        self.spectrum_mode_label.config(text=f"Spectrum: {self.spectrum_mode}")
        self._update_spectrum_only()

    def _update_spectrum_only(self):
        """Update only the spectrum plot (for mode toggle)."""
        data = self.fnirs.channel_data[self.current_channel]
        self.ax_spectrum.clear()
        self._plot_spectrum(data)

        # Re-apply styling
        self.ax_spectrum.set_facecolor('#1e1e1e')
        self.ax_spectrum.tick_params(colors='white')
        self.ax_spectrum.xaxis.label.set_color('white')
        self.ax_spectrum.yaxis.label.set_color('white')
        self.ax_spectrum.title.set_color('white')
        for spine in self.ax_spectrum.spines.values():
            spine.set_color('#555555')

        self.canvas.draw()

    def _update_plots(self):
        """Update all plots for the current channel."""
        # Update channel label
        self.channel_label.config(text=self._get_channel_text())

        # Get current channel data
        data = self.fnirs.channel_data[self.current_channel]
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

        # Re-apply styling after clear
        for ax in [self.ax_timeseries, self.ax_spectrogram, self.ax_spectrum]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#555555')

        self.canvas.draw()

    def _plot_timeseries(self, time: np.ndarray, data: np.ndarray):
        """Plot the timeseries with event markers."""
        self.ax_timeseries.plot(time, data, color='#00ff88', linewidth=0.8)
        self.ax_timeseries.set_xlabel('Time (s)')
        self.ax_timeseries.set_ylabel('Amplitude')
        self.ax_timeseries.set_title('Time Series')
        self.ax_timeseries.grid(True, alpha=0.3, color='#555555')

        # Add feature/event markers if available
        if self.fnirs.feature_onsets is not None and len(self.fnirs.feature_onsets) > 0:
            ymin, ymax = self.ax_timeseries.get_ylim()
            for onset in self.fnirs.feature_onsets:
                if 0 <= onset <= time[-1]:
                    self.ax_timeseries.axvline(x=onset, color='#ff6b6b', linestyle='--', alpha=0.7, linewidth=1)

        self.ax_timeseries.set_xlim(0, time[-1])

    def _plot_spectrogram(self, data: np.ndarray):
        """Plot the spectrogram of the signal."""
        # Compute spectrogram
        nperseg = min(256, len(data) // 4)
        if nperseg < 4:
            nperseg = len(data)

        f, t, Sxx = spectrogram(data, fs=self.fs, nperseg=nperseg, noverlap=nperseg // 2)

        # Limit frequency range for fNIRS (typically low frequency signals)
        freq_limit = min(self.fs / 2, 1.0)  # Show up to 1 Hz or Nyquist
        freq_mask = f <= freq_limit

        # Plot with log scale for better visibility
        Sxx_plot = Sxx[freq_mask, :]
        if Sxx_plot.max() > 0:
            Sxx_db = 10 * np.log10(Sxx_plot + 1e-12)
        else:
            Sxx_db = Sxx_plot

        im = self.ax_spectrogram.pcolormesh(t, f[freq_mask], Sxx_db, shading='gouraud', cmap='viridis')
        self.ax_spectrogram.set_xlabel('Time (s)')
        self.ax_spectrogram.set_ylabel('Frequency (Hz)')
        self.ax_spectrogram.set_title('Spectrogram')

    def _plot_spectrum(self, data: np.ndarray):
        """Plot the frequency spectrum (PSD or FFT based on current mode)."""
        freq_limit = min(self.fs / 2, 1.0)  # fNIRS signals are typically < 1 Hz

        if self.spectrum_mode == "PSD":
            freqs, spectrum = compute_psd(data, self.fs, freq_limit)
            self.ax_spectrum.semilogy(freqs, spectrum, color='#4ecdc4', linewidth=1.2)
            self.ax_spectrum.set_ylabel('Power Spectral Density')
            self.ax_spectrum.set_title('Frequency Spectrum (PSD)')
        else:  # FFT mode
            freqs, spectrum = compute_fft(data, self.fs, freq_limit)
            self.ax_spectrum.plot(freqs, spectrum, color='#ff9f43', linewidth=1.2)
            self.ax_spectrum.set_ylabel('Amplitude')
            self.ax_spectrum.set_title('Frequency Spectrum (FFT)')

        self.ax_spectrum.set_xlabel('Frequency (Hz)')
        self.ax_spectrum.grid(True, alpha=0.3, color='#555555')
        self.ax_spectrum.set_xlim(0, freq_limit)

    def run(self):
        """Start the GUI event loop."""
        self.root.mainloop()


def open(data: Union[str, fNIRS]):
    """
    Open the interactive visualizer for fNIRS data.

    Args:
        data: Either a path to a SNIRF file (str) or an fNIRS object

    Example:
        >>> import neuropipeline.visualizer as visualizer
        >>> visualizer.open("path/to/file.snirf")

        >>> from neuropipeline.fnirs import fNIRS
        >>> fnirs_data = fNIRS("path/to/file.snirf")
        >>> visualizer.open(fnirs_data)
    """
    viz = SNIRFVisualizer(data)
    viz.run()

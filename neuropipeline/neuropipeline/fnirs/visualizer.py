"""
Interactive GUI visualizer for fNIRS (SNIRF) data.

Usage:
    from neuropipeline.fnirs import visualizer
    visualizer.open(snirf_path_or_object)
"""
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.signal import spectrogram
import pywt  # Replaced scipy wavelet imports
from typing import Union

from .fnirs import fNIRS, compute_psd, compute_fft
from .analysis import complex_morlet_transform

# Marker color palette for distinguishing different marker types
MARKER_COLORS = [
    '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000'
]

# Module-level configuration
_config = {
    'spectrogram_freq_min': 0.0,
    'spectrogram_freq_max': None,
    'marker_dictionary': {},
    'spectrum_mode': 'FFT',
    'spectrogram_method': 'Wavelet',
}


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
        self.fs = self.fnirs.sampling_frequency

        # Spectrum display mode: "PSD" or "FFT" (read from module config)
        self.spectrum_mode = _config['spectrum_mode']

        # Spectrogram method: "STFT" or "Wavelet" (read from module config)
        self.spectrogram_method = _config['spectrogram_method']

        # Split into HbO and HbR channels
        self._split_channels()

        # Display options for HbO/HbR
        self.show_hbo = True
        self.show_hbr = True

        # Spectrogram frequency limits (read from module config)
        self.spectrogram_freq_min = _config['spectrogram_freq_min']
        self.spectrogram_freq_max = _config['spectrogram_freq_max']

        # Marker visibility and color mapping
        self._setup_markers()

        self._setup_gui()

    def _split_channels(self):
        """Split channel data into paired HbO and HbR arrays."""
        hbo_data, hbo_names, hbr_data, hbr_names = self.fnirs.split()

        # Store paired channel data
        self.hbo_data = hbo_data
        self.hbr_data = hbr_data
        self.hbo_names = hbo_names
        self.hbr_names = hbr_names

        # Number of paired channels (source-detector pairs)
        self.num_channels = len(hbo_names)

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

        if self.fnirs.feature_descriptions is not None and len(self.fnirs.feature_descriptions) > 0:
            # Get unique marker types
            self.marker_types = sorted(set(self.fnirs.feature_descriptions))

            # Assign colors, labels, and initialize visibility
            for i, marker_type in enumerate(self.marker_types):
                color_idx = i % len(MARKER_COLORS)
                self.marker_colors[marker_type] = MARKER_COLORS[color_idx]
                self.marker_visibility[marker_type] = True
                # Use custom label if available, otherwise default to "Marker {index}"
                self.marker_labels[marker_type] = marker_dict.get(marker_type, f"Marker {marker_type}")

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

        # HbO/HbR checkboxes
        checkbox_frame = ttk.Frame(parent)
        checkbox_frame.pack(fill=tk.X, pady=(0, 5))

        # HbO checkbox
        self.hbo_var = tk.BooleanVar(value=True)
        self.hbo_checkbox = tk.Checkbutton(
            checkbox_frame,
            text="HbO",
            variable=self.hbo_var,
            command=self._on_checkbox_change,
            bg='#2b2b2b',
            fg='#ff6b6b',
            selectcolor='#404040',
            activebackground='#2b2b2b',
            activeforeground='#ff6b6b',
            font=('Segoe UI', 11, 'bold'),
            cursor='hand2'
        )
        self.hbo_checkbox.pack(side=tk.LEFT, padx=(20, 10))

        # HbR checkbox
        self.hbr_var = tk.BooleanVar(value=True)
        self.hbr_checkbox = tk.Checkbutton(
            checkbox_frame,
            text="HbR",
            variable=self.hbr_var,
            command=self._on_checkbox_change,
            bg='#2b2b2b',
            fg='#4a90d9',
            selectcolor='#404040',
            activebackground='#2b2b2b',
            activeforeground='#4a90d9',
            font=('Segoe UI', 11, 'bold'),
            cursor='hand2'
        )
        self.hbr_checkbox.pack(side=tk.LEFT, padx=10)

        # Spectrum mode dropdown
        spectrum_label = ttk.Label(checkbox_frame, text="Spectrum:", font=('Segoe UI', 11))
        spectrum_label.pack(side=tk.LEFT, padx=(20, 5))

        self.spectrum_mode_var = tk.StringVar(value=self.spectrum_mode)
        self.spectrum_dropdown = ttk.Combobox(
            checkbox_frame,
            textvariable=self.spectrum_mode_var,
            values=["FFT", "PSD"],
            state="readonly",
            width=8,
            font=('Segoe UI', 10)
        )
        self.spectrum_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        self.spectrum_dropdown.bind("<<ComboboxSelected>>", self._on_spectrum_dropdown_change)

        # Spectrogram method dropdown
        spectrogram_label = ttk.Label(checkbox_frame, text="Spectrogram:", font=('Segoe UI', 11))
        spectrogram_label.pack(side=tk.LEFT, padx=(10, 5))

        self.spectrogram_method_var = tk.StringVar(value=self.spectrogram_method)
        self.spectrogram_dropdown = ttk.Combobox(
            checkbox_frame,
            textvariable=self.spectrogram_method_var,
            values=["Wavelet", "STFT", "CMT"],
            state="readonly",
            width=8,
            font=('Segoe UI', 10)
        )
        self.spectrogram_dropdown.pack(side=tk.LEFT, padx=(0, 20))
        self.spectrogram_dropdown.bind("<<ComboboxSelected>>", self._on_spectrogram_dropdown_change)

        # Marker toggle frame (only if markers exist)
        if len(self.marker_types) > 0:
            marker_outer_frame = ttk.Frame(parent)
            marker_outer_frame.pack(fill=tk.X, pady=(0, 5))

            marker_label = ttk.Label(marker_outer_frame, text="Markers:", font=('Segoe UI', 11))
            marker_label.pack(side=tk.LEFT, padx=(20, 10))

            # Create a canvas with horizontal scrolling for many markers
            marker_canvas = tk.Canvas(
                marker_outer_frame,
                bg='#2b2b2b',
                highlightthickness=0,
                height=30
            )
            marker_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Inner frame to hold checkboxes
            marker_inner_frame = tk.Frame(marker_canvas, bg='#2b2b2b')
            marker_canvas.create_window((0, 0), window=marker_inner_frame, anchor='nw')

            for marker_type in self.marker_types:
                color = self.marker_colors[marker_type]
                label = self.marker_labels[marker_type]
                var = tk.BooleanVar(value=True)
                self.marker_vars[marker_type] = var

                cb = tk.Checkbutton(
                    marker_inner_frame,
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

            # Update scroll region when frame is configured
            def update_scroll_region(event):
                marker_canvas.configure(scrollregion=marker_canvas.bbox('all'))

            marker_inner_frame.bind('<Configure>', update_scroll_region)

            # Enable horizontal scrolling with mouse wheel (Shift+Scroll or horizontal scroll)
            def on_mouse_wheel(event):
                marker_canvas.xview_scroll(int(-1 * (event.delta / 120)), 'units')

            marker_canvas.bind('<Shift-MouseWheel>', on_mouse_wheel)
            marker_canvas.bind('<MouseWheel>', on_mouse_wheel)

    def _create_plot_area(self, parent):
        """Create the matplotlib plot area with scrollable canvas."""
        # Create a canvas with scrollbar for the plots
        plot_container = ttk.Frame(parent)
        plot_container.pack(fill=tk.BOTH, expand=True)

        # Create matplotlib figure - will add subplots dynamically
        self.fig = Figure(figsize=(12, 10), facecolor='#2b2b2b')
        self.plot_container = plot_container

        # Initialize axes as None - will be created in _setup_axes
        self.ax_timeseries = None
        self.ax_spectrogram_hbo = None
        self.ax_spectrogram_hbr = None
        self.ax_spectrum = None

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
        """Setup axes based on current HbO/HbR display settings."""
        self.fig.clear()

        # Determine layout based on what's being shown
        both_shown = self.show_hbo and self.show_hbr

        # Leave space on right for legend (0.88 instead of 0.95)
        if both_shown:
            # 3 rows: timeseries, spectrograms (side by side), spectrum
            gs = self.fig.add_gridspec(3, 2, hspace=0.4, wspace=0.15,
                                       left=0.06, right=0.88, top=0.95, bottom=0.08)
            self.ax_timeseries = self.fig.add_subplot(gs[0, :])  # Full width
            self.ax_spectrogram_hbo = self.fig.add_subplot(gs[1, 0])  # Left
            self.ax_spectrogram_hbr = self.fig.add_subplot(gs[1, 1])  # Right
            self.ax_spectrum = self.fig.add_subplot(gs[2, :])  # Full width
            axes_list = [self.ax_timeseries, self.ax_spectrogram_hbo, self.ax_spectrogram_hbr, self.ax_spectrum]
        else:
            # 3 rows: timeseries, single spectrogram, spectrum
            gs = self.fig.add_gridspec(3, 1, hspace=0.4, left=0.06, right=0.88, top=0.95, bottom=0.08)
            self.ax_timeseries = self.fig.add_subplot(gs[0])
            self.ax_spectrogram_hbo = self.fig.add_subplot(gs[1])  # Use HbO slot for single spectrogram
            self.ax_spectrogram_hbr = None
            self.ax_spectrum = self.fig.add_subplot(gs[2])
            axes_list = [self.ax_timeseries, self.ax_spectrogram_hbo, self.ax_spectrum]

        # Style all axes
        for ax in axes_list:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#555555')

    def _get_channel_text(self) -> str:
        """Get the current channel display text."""
        channel_name = self.hbo_names[self.current_channel]
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

    def _on_checkbox_change(self):
        """Handle HbO/HbR checkbox changes."""
        old_both = self.show_hbo and self.show_hbr
        self.show_hbo = self.hbo_var.get()
        self.show_hbr = self.hbr_var.get()
        new_both = self.show_hbo and self.show_hbr

        # Rebuild axes if layout needs to change (both vs single)
        if old_both != new_both:
            self._setup_axes()

        self._update_plots()

    def _on_marker_toggle(self, marker_type):
        """Handle marker visibility toggle."""
        self.marker_visibility[marker_type] = self.marker_vars[marker_type].get()
        self._update_timeseries_only()

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

    def _toggle_spectrogram_method(self):
        """Toggle between STFT and Wavelet spectrogram methods."""
        self.spectrogram_method = "Wavelet" if self.spectrogram_method == "STFT" else "STFT"
        self.spectrogram_method_var.set(self.spectrogram_method)
        self._update_spectrogram_only()

    def _update_spectrogram_only(self):
        """Update only the spectrogram plot(s) (for method toggle)."""
        hbo_data = self.hbo_data[self.current_channel]
        hbr_data = self.hbr_data[self.current_channel]

        # Clear spectrogram axes
        self.ax_spectrogram_hbo.clear()
        if self.ax_spectrogram_hbr is not None:
            self.ax_spectrogram_hbr.clear()

        # Replot spectrograms
        self._plot_spectrogram(hbo_data, hbr_data)

        # Re-apply styling
        self._style_axis(self.ax_spectrogram_hbo)
        if self.ax_spectrogram_hbr is not None:
            self._style_axis(self.ax_spectrogram_hbr)

        self.canvas.draw()

    def _update_spectrum_only(self):
        """Update only the spectrum plot (for mode toggle)."""
        hbo_data = self.hbo_data[self.current_channel]
        hbr_data = self.hbr_data[self.current_channel]
        self.ax_spectrum.clear()
        self._plot_spectrum(hbo_data, hbr_data)

        # Re-apply styling
        self._style_axis(self.ax_spectrum)
        self.canvas.draw()

    def _update_timeseries_only(self):
        """Update only the timeseries plot (for marker toggle)."""
        hbo_data = self.hbo_data[self.current_channel]
        hbr_data = self.hbr_data[self.current_channel]
        time = np.arange(len(hbo_data)) / self.fs

        # Save current axis limits to preserve zoom/pan
        xlim = self.ax_timeseries.get_xlim()
        ylim = self.ax_timeseries.get_ylim()

        self.ax_timeseries.clear()
        self._plot_timeseries(time, hbo_data, hbr_data)

        # Restore axis limits
        self.ax_timeseries.set_xlim(xlim)
        self.ax_timeseries.set_ylim(ylim)

        # Re-apply styling
        self._style_axis(self.ax_timeseries)
        self.canvas.draw()

    def _style_axis(self, ax):
        """Apply dark theme styling to an axis."""
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#555555')

    def _update_plots(self):
        """Update all plots for the current channel."""
        # Update channel label
        self.channel_label.config(text=self._get_channel_text())

        # Get current channel HbO and HbR data
        hbo_data = self.hbo_data[self.current_channel]
        hbr_data = self.hbr_data[self.current_channel]
        time = np.arange(len(hbo_data)) / self.fs

        # Build list of axes to clear/style
        axes_list = [self.ax_timeseries, self.ax_spectrogram_hbo, self.ax_spectrum]
        if self.ax_spectrogram_hbr is not None:
            axes_list.append(self.ax_spectrogram_hbr)

        # Clear all axes
        for ax in axes_list:
            ax.clear()

        # Plot timeseries
        self._plot_timeseries(time, hbo_data, hbr_data)

        # Plot spectrogram(s)
        self._plot_spectrogram(hbo_data, hbr_data)

        # Plot frequency spectrum
        self._plot_spectrum(hbo_data, hbr_data)

        # Re-apply styling after clear
        for ax in axes_list:
            self._style_axis(ax)

        self.canvas.draw()
    def _plot_cmt_spectrogram(self, ax, data: np.ndarray, title: str, cmap: str):
        """Plot Complex Morlet Transform spectrogram on the given axis."""
        # Determine frequency limits (same as other spectrogram methods)
        freq_min = self.spectrogram_freq_min
        freq_max = self.spectrogram_freq_max if self.spectrogram_freq_max is not None else min(self.fs / 2, 1.0)

        # Generate target frequencies for the CMT
        num_freqs = 64
        frequencies = np.linspace(max(freq_min, 0.001), freq_max, num_freqs)

        # Convert frequencies to scales for PyWavelets
        wavelet_name = 'cmor1.5-1.0'
        sampling_period = 1.0 / self.fs
        scales = pywt.frequency2scale(wavelet_name, frequencies * sampling_period)

        # Compute the CWT with proper sampling period
        coefficients, freqs_out = pywt.cwt(data, scales, wavelet_name, sampling_period=sampling_period)

        # Compute magnitude
        magnitude = np.abs(coefficients)

        # Create time axis
        t = np.arange(len(data)) / self.fs

        # Plot using frequencies on y-axis
        ax.pcolormesh(t, freqs_out, magnitude, shading='gouraud', cmap=cmap)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f"{title} (CMT)")
    
    def _plot_timeseries(self, time: np.ndarray, hbo_data: np.ndarray, hbr_data: np.ndarray):
        """Plot the timeseries with event markers for HbO and HbR."""
        if self.show_hbo:
            self.ax_timeseries.plot(time, hbo_data, color='#ff6b6b', linewidth=0.8, label='HbO')
        if self.show_hbr:
            self.ax_timeseries.plot(time, hbr_data, color='#4a90d9', linewidth=0.8, label='HbR')

        self.ax_timeseries.set_xlabel('Time (s)')
        self.ax_timeseries.set_ylabel('Amplitude')
        self.ax_timeseries.set_title('Time Series')
        self.ax_timeseries.grid(True, alpha=0.3, color='#555555')

        # Add feature/event markers if available
        marker_handles = []
        marker_legend_labels = []
        if self.fnirs.feature_onsets is not None and len(self.fnirs.feature_onsets) > 0:
            # Track which marker types have been added to legend
            added_to_legend = set()

            for i, onset in enumerate(self.fnirs.feature_onsets):
                if 0 <= onset <= time[-1]:
                    marker_type = self.fnirs.feature_descriptions[i]

                    # Skip if this marker type is hidden
                    if not self.marker_visibility.get(marker_type, True):
                        continue

                    color = self.marker_colors.get(marker_type, '#ffcc00')
                    line = self.ax_timeseries.axvline(
                        x=onset, color=color, linestyle='--', alpha=0.7, linewidth=1
                    )

                    # Add to legend only once per marker type
                    if marker_type not in added_to_legend:
                        marker_handles.append(line)
                        # Use custom label if available
                        label = self.marker_labels.get(marker_type, f"Marker {marker_type}")
                        marker_legend_labels.append(label)
                        added_to_legend.add(marker_type)

        # Build combined legend (signals + markers)
        handles, labels = [], []
        if self.show_hbo or self.show_hbr:
            ax_handles, ax_labels = self.ax_timeseries.get_legend_handles_labels()
            handles.extend(ax_handles)
            labels.extend(ax_labels)

        handles.extend(marker_handles)
        labels.extend(marker_legend_labels)

        if handles:
            # Place legend outside the plot area (to the right) to avoid obstructing data
            self.ax_timeseries.legend(
                handles, labels,
                loc='upper left',
                bbox_to_anchor=(1.01, 1),
                facecolor='#2b2b2b',
                edgecolor='#555555',
                labelcolor='white',
                fontsize=8
            )

        self.ax_timeseries.set_xlim(0, time[-1])

    def _plot_spectrogram(self, hbo_data: np.ndarray, hbr_data: np.ndarray):
        """Plot the spectrogram(s) of the signal(s)."""
        both_shown = self.show_hbo and self.show_hbr

        if both_shown:
            # Plot separate spectrograms for HbO and HbR
            self._plot_single_spectrogram(self.ax_spectrogram_hbo, hbo_data, 'Spectrogram (HbO)', 'viridis')
            self._plot_single_spectrogram(self.ax_spectrogram_hbr, hbr_data, 'Spectrogram (HbR)', 'viridis')
        elif self.show_hbo:
            self._plot_single_spectrogram(self.ax_spectrogram_hbo, hbo_data, 'Spectrogram (HbO)', 'viridis')
        elif self.show_hbr:
            self._plot_single_spectrogram(self.ax_spectrogram_hbo, hbr_data, 'Spectrogram (HbR)', 'viridis')
        else:
            # Neither shown - just show empty plot
            self.ax_spectrogram_hbo.set_title('Spectrogram')
            self.ax_spectrogram_hbo.set_xlabel('Time (s)')
            self.ax_spectrogram_hbo.set_ylabel('Frequency (Hz)')

    def _plot_single_spectrogram(self, ax, data: np.ndarray, title: str, cmap: str):
        """Plot a single spectrogram on the given axis."""
        # Determine frequency limits
        freq_min = self.spectrogram_freq_min
        freq_max = self.spectrogram_freq_max if self.spectrogram_freq_max is not None else min(self.fs / 2, 1.0)

        if self.spectrogram_method == "STFT":
            self._plot_stft_spectrogram(ax, data, title, cmap, freq_min, freq_max)
        elif self.spectrogram_method == "CMT":
            self._plot_cmt_spectrogram(ax, data, title, cmap)
        else:  # Wavelet
            self._plot_wavelet_spectrogram(ax, data, title, cmap, freq_min, freq_max)

    def _plot_stft_spectrogram(self, ax, data: np.ndarray, title: str, cmap: str,
                                freq_min: float, freq_max: float):
        """Plot STFT-based spectrogram."""
        nperseg = min(256, len(data) // 4)
        if nperseg < 4:
            nperseg = len(data)

        f, t, Sxx = spectrogram(data, fs=self.fs, nperseg=nperseg, noverlap=nperseg // 2)
        # Apply frequency mask
        freq_mask = (f >= freq_min) & (f <= freq_max)

        # Plot with log scale for better visibility
        Sxx_plot = Sxx[freq_mask, :]
        if Sxx_plot.max() > 0:
            Sxx_db = 10 * np.log10(Sxx_plot + 1e-12)
        else:
            Sxx_db = Sxx_plot

        ax.pcolormesh(t, f[freq_mask], Sxx_db, shading='gouraud', cmap=cmap)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f"{title} (STFT)")

    def _plot_wavelet_spectrogram(self, ax, data: np.ndarray, title: str, cmap: str,
                                   freq_min: float, freq_max: float):
        """Plot wavelet-based spectrogram using continuous wavelet transform."""
        # Define the wavelet to use (Complex Morlet is the equivalent of morlet2)
        # 'cmor1.5-1.0' is a common choice for fNIRS: center freq 1.5, bandwidth 1.0
        wavelet_name = 'cmor1.5-1.0'
        
        # Generate frequencies for wavelet analysis
        num_freqs = 64
        frequencies = np.linspace(max(freq_min, 0.001), freq_max, num_freqs)

        # Convert frequencies to scales for PyWavelets
        # PyWavelets: scale = center_frequency / (target_frequency / sampling_rate)
        sampling_period = 1.0 / self.fs
        scales = pywt.frequency2scale(wavelet_name, frequencies * sampling_period)

        # Compute CWT 
        # Returns: [coefficients, frequencies]
        coef, freqs_out = pywt.cwt(data, scales, wavelet_name, sampling_period=sampling_period)
        
        power = np.abs(coef) ** 2

        # Create time axis
        t = np.arange(len(data)) / self.fs

        # Plot with log scale for better visibility
        if power.max() > 0:
            # We use freqs_out to ensure the Y-axis matches the actual computed frequencies
            power_db = 10 * np.log10(power + 1e-12)
        else:
            power_db = power

        # Use pcolormesh to display the power
        im = ax.pcolormesh(t, freqs_out, power_db, shading='gouraud', cmap=cmap)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f"{title} (Wavelet)")

    def _plot_spectrum(self, hbo_data: np.ndarray, hbr_data: np.ndarray):
        """Plot the frequency spectrum (PSD or FFT based on current mode) for HbO and HbR."""
        freq_limit = min(self.fs / 2, 1.0)  # fNIRS signals are typically < 1 Hz

        if self.spectrum_mode == "PSD":
            if self.show_hbo:
                freqs, spectrum = compute_psd(hbo_data, self.fs, freq_limit)
                self.ax_spectrum.semilogy(freqs, spectrum, color='#ff6b6b', linewidth=1.2, label='HbO')
            if self.show_hbr:
                freqs, spectrum = compute_psd(hbr_data, self.fs, freq_limit)
                self.ax_spectrum.semilogy(freqs, spectrum, color='#4a90d9', linewidth=1.2, label='HbR')
            self.ax_spectrum.set_ylabel('Power Spectral Density')
            self.ax_spectrum.set_title('Frequency Spectrum (PSD)')
        else:  # FFT mode
            if self.show_hbo:
                freqs, spectrum = compute_fft(hbo_data, self.fs, freq_limit)
                self.ax_spectrum.plot(freqs, spectrum, color='#ff6b6b', linewidth=1.2, label='HbO')
            if self.show_hbr:
                freqs, spectrum = compute_fft(hbr_data, self.fs, freq_limit)
                self.ax_spectrum.plot(freqs, spectrum, color='#4a90d9', linewidth=1.2, label='HbR')
            self.ax_spectrum.set_ylabel('Amplitude')
            self.ax_spectrum.set_title('Frequency Spectrum (FFT)')

        self.ax_spectrum.set_xlabel('Frequency (Hz)')
        self.ax_spectrum.grid(True, alpha=0.3, color='#555555')
        self.ax_spectrum.set_xlim(0, freq_limit)

        # Add legend if at least one signal is shown
        if self.show_hbo or self.show_hbr:
            self.ax_spectrum.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='#555555', labelcolor='white')

    def set_spectrogram_limits(self, freq_min: float = 0.0, freq_max: float = None):
        """
        Set the frequency limits for the spectrogram display.

        Args:
            freq_min: Minimum frequency to display (Hz). Default is 0.0.
            freq_max: Maximum frequency to display (Hz). Default is None,
                      which uses min(sampling_rate/2, 1.0).
        """
        self.spectrogram_freq_min = freq_min
        self.spectrogram_freq_max = freq_max
        self._update_plots()

    def run(self):
        """Start the GUI event loop."""
        self.root.mainloop()


def set_spectrogram_limits(freq_min: float = 0.0, freq_max: float = None):
    """
    Configure the spectrogram frequency limits for the visualizer.

    Call this before open() to set the frequency range displayed in spectrograms.

    Args:
        freq_min: Minimum frequency to display (Hz). Default is 0.0.
        freq_max: Maximum frequency to display (Hz). Default is None,
                  which uses min(sampling_rate/2, 1.0).

    Example:
        >>> import neuropipeline.visualizer as nplv
        >>> nplv.set_spectrogram_limits(0.01, 0.1)
        >>> nplv.open(fnirs)
    """
    _config['spectrogram_freq_min'] = freq_min
    _config['spectrogram_freq_max'] = freq_max


def set_marker_dictionary(marker_dict: dict):
    """
    Configure custom labels for marker types.

    Call this before open() to set human-readable names for marker indices.
    Marker types not in the dictionary will display as "Marker {index}".

    Args:
        marker_dict: Dictionary mapping marker index (int) to label (str).

    Example:
        >>> import neuropipeline.visualizer as nplv
        >>> nplv.set_marker_dictionary({1: "Rest", 3: "Pronation", 4: "Supination"})
        >>> nplv.open(fnirs)
    """
    _config['marker_dictionary'] = marker_dict


def set_spectrum_mode(mode: str = "FFT"):
    """
    Configure the default spectrum display mode.

    Call this before open() to set whether to display FFT or PSD.

    Args:
        mode: Either "FFT" or "PSD". Default is "FFT".

    Example:
        >>> from neuropipeline.fnirs import visualizer
        >>> visualizer.set_spectrum_mode("PSD")
        >>> visualizer.open(fnirs)
    """
    if mode not in ("FFT", "PSD"):
        raise ValueError("mode must be 'FFT' or 'PSD'")
    _config['spectrum_mode'] = mode


def set_spectrogram_method(method: str = "Wavelet"):
    """
    Configure the default spectrogram computation method.

    Call this before open() to set whether to use STFT, Wavelet, or CMT.
    Note: Wavelet and CMT are more computationally expensive but provide better
    time-frequency resolution. Use STFT for faster visualization.

    Args:
        method: "STFT", "Wavelet", or "CMT" (Complex Morlet Transform). Default is "Wavelet".

    Example:
        >>> from neuropipeline.fnirs import visualizer
        >>> visualizer.set_spectrogram_method("STFT")  # Faster
        >>> visualizer.set_spectrogram_method("CMT")   # Complex Morlet Transform
        >>> visualizer.open(fnirs)
    """
    if method not in ("STFT", "Wavelet", "CMT"):
        raise ValueError("method must be 'STFT', 'Wavelet', or 'CMT'")
    _config['spectrogram_method'] = method


def open(data: Union[str, fNIRS]):
    """
    Open the interactive visualizer for fNIRS data.

    Args:
        data: Either a path to a SNIRF file (str) or an fNIRS object

    Example:
        >>> from neuropipeline.fnirs import visualizer
        >>> visualizer.open("path/to/file.snirf")

        >>> # With custom settings
        >>> visualizer.set_spectrogram_method("STFT")  # Faster than Wavelet
        >>> visualizer.set_spectrum_mode("PSD")
        >>> visualizer.set_spectrogram_limits(0.01, 0.5)
        >>> visualizer.open("path/to/file.snirf")
    """
    viz = SNIRFVisualizer(data)
    viz.run()

# visualizer_qt.py

import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QLabel,
    QPushButton,
    QCheckBox,
    QFileDialog,
    QFrame,
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .fnirs.fnirs import fNIRS
from .fnirs.preprocessor import fNIRSPreprocessor

# Marker color palette for distinguishing different marker types
MARKER_COLORS = [
    '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000'
]


class TimeSeriesWidget(QWidget):
    """Time series visualization widget with matplotlib canvas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fnirs = None
        self.current_channel = 0
        self.show_hbo = True
        self.show_hbr = True

        # Channel data after splitting
        self.hbo_data = None
        self.hbr_data = None
        self.hbo_names = []
        self.hbr_names = []
        self.num_channels = 0
        self.fs = 1.0

        # Marker settings
        self.marker_colors = {}
        self.marker_visibility = {}
        self.marker_labels = {}

        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Controls frame
        controls_frame = QFrame()
        controls_frame.setStyleSheet("QFrame { background-color: #2b2b2b; }")
        controls_layout = QHBoxLayout(controls_frame)

        # Previous button
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedWidth(50)
        self.prev_btn.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                color: white;
                border: none;
                font-size: 16px;
                padding: 5px;
            }
            QPushButton:hover { background-color: #505050; }
            QPushButton:disabled { background-color: #303030; color: #666666; }
        """)
        self.prev_btn.clicked.connect(self._prev_channel)
        controls_layout.addWidget(self.prev_btn)

        # Channel label
        self.channel_label = QLabel("No data loaded")
        self.channel_label.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        self.channel_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self.channel_label, 1)

        # Next button
        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedWidth(50)
        self.next_btn.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                color: white;
                border: none;
                font-size: 16px;
                padding: 5px;
            }
            QPushButton:hover { background-color: #505050; }
            QPushButton:disabled { background-color: #303030; color: #666666; }
        """)
        self.next_btn.clicked.connect(self._next_channel)
        controls_layout.addWidget(self.next_btn)

        layout.addWidget(controls_frame)

        # Channel slider
        slider_frame = QFrame()
        slider_frame.setStyleSheet("QFrame { background-color: #2b2b2b; }")
        slider_layout = QHBoxLayout(slider_frame)

        self.channel_slider = QSlider(Qt.Orientation.Horizontal)
        self.channel_slider.setMinimum(1)
        self.channel_slider.setMaximum(1)
        self.channel_slider.setValue(1)
        self.channel_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #404040;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a90d9;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """)
        self.channel_slider.valueChanged.connect(self._on_slider_change)
        slider_layout.addWidget(self.channel_slider)

        layout.addWidget(slider_frame)

        # HbO/HbR checkboxes
        checkbox_frame = QFrame()
        checkbox_frame.setStyleSheet("QFrame { background-color: #2b2b2b; }")
        checkbox_layout = QHBoxLayout(checkbox_frame)

        self.hbo_checkbox = QCheckBox("HbO")
        self.hbo_checkbox.setChecked(True)
        self.hbo_checkbox.setStyleSheet("QCheckBox { color: #ff6b6b; font-weight: bold; }")
        self.hbo_checkbox.stateChanged.connect(self._on_checkbox_change)
        checkbox_layout.addWidget(self.hbo_checkbox)

        self.hbr_checkbox = QCheckBox("HbR")
        self.hbr_checkbox.setChecked(True)
        self.hbr_checkbox.setStyleSheet("QCheckBox { color: #4a90d9; font-weight: bold; }")
        self.hbr_checkbox.stateChanged.connect(self._on_checkbox_change)
        checkbox_layout.addWidget(self.hbr_checkbox)

        checkbox_layout.addStretch()
        layout.addWidget(checkbox_frame)

        # Matplotlib figure and canvas
        self.fig = Figure(figsize=(10, 4), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style_axis(self.ax)

        layout.addWidget(self.canvas, 1)

        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # Disable buttons initially
        self._update_button_states()

    def _style_axis(self, ax):
        """Apply dark theme styling to axis."""
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#555555')

    def load_data(self, fnirs_data):
        """Load fNIRS data into the widget."""
        self.fnirs = fnirs_data
        self.fs = fnirs_data.sampling_frequency

        # Split channels into HbO and HbR
        hbo_data, hbo_names, hbr_data, hbr_names = fnirs_data.split()
        self.hbo_data = hbo_data
        self.hbr_data = hbr_data
        self.hbo_names = hbo_names
        self.hbr_names = hbr_names
        self.num_channels = len(hbo_names)

        # Setup markers
        self._setup_markers()

        # Update slider range
        self.channel_slider.setMaximum(self.num_channels)
        self.channel_slider.setValue(1)
        self.current_channel = 0

        # Update display
        self._update_button_states()
        self._update_plot()

    def _setup_markers(self):
        """Setup marker types, colors, and visibility."""
        self.marker_colors = {}
        self.marker_visibility = {}
        self.marker_labels = {}

        if self.fnirs.feature_descriptions is not None and len(self.fnirs.feature_descriptions) > 0:
            marker_types = sorted(set(self.fnirs.feature_descriptions))
            for i, marker_type in enumerate(marker_types):
                color_idx = i % len(MARKER_COLORS)
                self.marker_colors[marker_type] = MARKER_COLORS[color_idx]
                self.marker_visibility[marker_type] = True
                self.marker_labels[marker_type] = f"Marker {marker_type}"

    def _update_button_states(self):
        """Update navigation button enabled states."""
        self.prev_btn.setEnabled(self.current_channel > 0)
        self.next_btn.setEnabled(self.current_channel < self.num_channels - 1)

    def _get_channel_text(self):
        """Get current channel display text."""
        if self.num_channels == 0:
            return "No data loaded"
        channel_name = self.hbo_names[self.current_channel]
        return f"Channel {self.current_channel + 1}/{self.num_channels}: {channel_name}"

    def _prev_channel(self):
        """Navigate to previous channel."""
        if self.current_channel > 0:
            self.current_channel -= 1
            self.channel_slider.setValue(self.current_channel + 1)
            self._update_button_states()
            self._update_plot()

    def _next_channel(self):
        """Navigate to next channel."""
        if self.current_channel < self.num_channels - 1:
            self.current_channel += 1
            self.channel_slider.setValue(self.current_channel + 1)
            self._update_button_states()
            self._update_plot()

    def _on_slider_change(self, value):
        """Handle slider value change."""
        new_channel = value - 1
        if new_channel != self.current_channel and 0 <= new_channel < self.num_channels:
            self.current_channel = new_channel
            self._update_button_states()
            self._update_plot()

    def _on_checkbox_change(self):
        """Handle HbO/HbR checkbox changes."""
        self.show_hbo = self.hbo_checkbox.isChecked()
        self.show_hbr = self.hbr_checkbox.isChecked()
        self._update_plot()

    def _update_plot(self):
        """Update the time series plot."""
        self.channel_label.setText(self._get_channel_text())

        if self.hbo_data is None or self.num_channels == 0:
            return

        self.ax.clear()

        hbo_data = self.hbo_data[self.current_channel]
        hbr_data = self.hbr_data[self.current_channel]
        time = np.arange(len(hbo_data)) / self.fs

        # Plot time series
        if self.show_hbo:
            self.ax.plot(time, hbo_data, color='#ff6b6b', linewidth=0.8, label='HbO')
        if self.show_hbr:
            self.ax.plot(time, hbr_data, color='#4a90d9', linewidth=0.8, label='HbR')

        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Time Series')
        self.ax.grid(True, alpha=0.3, color='#555555')

        # Add event markers
        marker_handles = []
        marker_legend_labels = []
        if self.fnirs.feature_onsets is not None and len(self.fnirs.feature_onsets) > 0:
            added_to_legend = set()

            for i, onset in enumerate(self.fnirs.feature_onsets):
                if 0 <= onset <= time[-1]:
                    marker_type = self.fnirs.feature_descriptions[i]

                    if not self.marker_visibility.get(marker_type, True):
                        continue

                    color = self.marker_colors.get(marker_type, '#ffcc00')
                    line = self.ax.axvline(x=onset, color=color, linestyle='--', alpha=0.7, linewidth=1)

                    if marker_type not in added_to_legend:
                        marker_handles.append(line)
                        label = self.marker_labels.get(marker_type, f"Marker {marker_type}")
                        marker_legend_labels.append(label)
                        added_to_legend.add(marker_type)

        # Build legend
        handles, labels = [], []
        if self.show_hbo or self.show_hbr:
            ax_handles, ax_labels = self.ax.get_legend_handles_labels()
            handles.extend(ax_handles)
            labels.extend(ax_labels)

        handles.extend(marker_handles)
        labels.extend(marker_legend_labels)

        if handles:
            self.ax.legend(handles, labels, loc='upper right',
                          facecolor='#2b2b2b', edgecolor='#555555', labelcolor='white')

        self.ax.set_xlim(0, time[-1])
        self._style_axis(self.ax)
        self.canvas.draw()


class FNIRSVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("fNIRS Visualizer")
        self.setMinimumSize(1000, 700)

        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QMenuBar { background-color: #2b2b2b; color: white; }
            QMenuBar::item:selected { background-color: #404040; }
            QMenu { background-color: #2b2b2b; color: white; }
            QMenu::item:selected { background-color: #404040; }
            QStatusBar { background-color: #2b2b2b; color: white; }
        """)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)
        self.layout.setContentsMargins(5, 5, 5, 5)

        # Menu bar
        self._setup_menu()

        # Preprocessor
        self.preprocessor_fnirs = fNIRSPreprocessor()

        # Time Series widget
        self.time_series_widget = TimeSeriesWidget()
        self.layout.addWidget(self.time_series_widget)

        # Status bar
        self.statusBar().showMessage("Ready - Open a .snirf file to begin")

        # TODO : Add more visualization components
        # 2. Topographical Map (For starter, we can just have a vertical list of Channel 1: S1-D1, ...)
        # 3. Spectrogram (STFT and Wavelet)
        # 4. Frequency Plot (FFT and PSD)

    def _setup_menu(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction("&Open...", self._on_open)
        file_menu.addSeparator()
        file_menu.addAction("&Exit", self.close)

        # View menu (placeholder for later)
        view_menu = menu_bar.addMenu("&View")
        view_menu.addAction("&Reset Layout", self._on_reset_layout)

        # Preprocessing menu
        preprocessing_menu = menu_bar.addMenu("&Preprocessing")
        preprocessing_menu.addAction("&Edit Preprocessing", self._on_edit_preprocessing)

    def _on_open(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open SNIRF File",
            "",
            "SNIRF Files (*.snirf);;All Files (*)"
        )

        if file_path:
            self.statusBar().showMessage(f"Loading {file_path}...")
            try:
                fnirs_data = fNIRS(file_path)
                self.time_series_widget.load_data(fnirs_data)
                self.statusBar().showMessage(f"Loaded: {file_path}")
            except Exception as e:
                self.statusBar().showMessage(f"Error loading file: {e}")

    def _on_reset_layout(self):
        self.statusBar().showMessage("Reset layout clicked (not implemented)")

    def _on_edit_preprocessing(self):
        self.statusBar().showMessage("Edit preprocessing clicked (not implemented)")


def main():
    app = QApplication(sys.argv)
    window = FNIRSVisualizer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

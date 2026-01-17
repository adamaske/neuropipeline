# Neuropipeline Visualizer

Interactive GUI for exploring fNIRS (SNIRF) data channel-by-channel.

## Usage

```python
import neuropipeline.visualizer as visualizer

# From file path
visualizer.open("path/to/file.snirf")

# From existing fNIRS object
from neuropipeline.fnirs import fNIRS
data = fNIRS("path/to/file.snirf")
visualizer.open(data)
```

## Features

**Channel Navigation**
- Header displays current channel name and index
- Arrow buttons or Left/Right keys to switch channels

**Time Series Plot**
- Full signal trace with event markers (red dashed lines)
- Zoomable and pannable via matplotlib toolbar

**Spectrogram**
- Time-frequency representation
- Frequency range limited to ~1 Hz (typical fNIRS band)

**Frequency Spectrum**
- Toggle between PSD and FFT views using arrows or Spacebar
- PSD: Log-scale power spectral density
- FFT: Linear amplitude for precise low-frequency inspection

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Left/Right | Previous/Next channel |
| Space | Toggle PSD/FFT |
| Escape | Close window |

## Requirements

Uses tkinter (built-in) and matplotlib for rendering. No additional dependencies beyond the standard neuropipeline requirements.

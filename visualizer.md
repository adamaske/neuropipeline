# Neuropipeline Visualizer

Interactive GUI for exploring fNIRS (SNIRF) data channel-by-channel.

## Usage

```python
from neuropipeline.fnirs import fNIRS
import neuropipeline.visualizer as nplv

fnirs = fNIRS("path/to/your_file.snirf")
fnirs.preprocess(
    optical_density=True,
    hemoglobin_concentration=True,
    temporal_filtering=True,
    normalization=False,
    detrending=True
)

nplv.open(fnirs)
```

## Example
These examples show single trials from a subject in my master thesis experiments. The time series, spectrogram and frequency (PSD, FFT) are plotted for each case. The red dashed lines represent markers in the data, specifically when each block starts (rest, task) as well as metadata markers describing the robot's movement and actions. The stimuli was delivered as 6 indentations and subsequent shears of the heel edge by a robot actuator. In the supination case (left), we see a clear correlation with increased HbO in the first two stimuli periods. Looking at the spectrogram (y: 0.00-0.10 Hz), the neurogenic band centered at 0.025 Hz shows activity throughout, with notable spikes coinciding with the HbO increases at stimuli onset. This is a strong indication of captured neurogenic activity in the sensory cortex resulting from mechanical stimulation of the lateral heel edge. The pronation case (right) does not exhibit the same pattern; neurogenic activity remains consistently low across the trial spectrogram, and while the time series shows some local peaks during task durations,  

<table>
  <tr>
    <td><img src="visualizer_example.jpg" alt="Supination case"/></td>
    <td><img src="visualizer_example_pronation.jpg" alt="Pronation case"/></td>
  </tr>
  <tr>
    <td align="center"><em>Supination</em></td>
    <td align="center"><em>Pronation</em></td>
  </tr>
</table>

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



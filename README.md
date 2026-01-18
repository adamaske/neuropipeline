# neuropipeline

`neuropipeline` is a tool for quick and easy to use preprocessing and visualization of Functional Near-Infrared Spectroscopy (fNIRS) data.  

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

snirf.write_snirf("path/to/your_new_file.snirf") # WARNING: Be cautious not to overwrite any data you want to keep. 
```
## Analysis Example: Heel Stimulation

These plots display data from a single subject during a robotic heel-stimulation experiment, showing the Time Series, Spectrogram, and Frequency (PSD/FFT) for two different scenarios. The vertical red dashed lines indicate "markers," which show exactly when a task started or when the robot moved. In this experiment, a robot stimulated the heel 6 times. In the Supination case (left), we see a clear success: oxygenated hemoglobin (HbO) increases right when the stimuli begin. This is supported by the spectrogram, where we see "spikes" of activity at 0.025 Hz (the neurogenic band) that align perfectly with the robot's movements. This confirms the pipeline has successfully captured brain activity in the sensory cortex. In contrast, the Pronation case (right) shows consistently low activity in the spectrogram, and while the time series has some small peaks, they do not show the same clear correlation with the stimulation.

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
## Installation

```bash
python -m pip install neuropipeline
```

## Advanced Usage

```python
fnirs = fNIRS("path/to/your_file.snirf")
    
if preprocess:
    fnirs.preprocess(optical_density=True,
                 hemoglobin_concentration=True,
                 motion_correction=True,
                 temporal_filtering=True,
                 detrending=True,
                 normalization=False
                 )
nplv.set_spectrogram_limits(0.0, 0.12) # Hz

nplv.set_marker_dictionary({
    0: "Rest",
    1: "Stimuli A",
    2: "Stimuli B",
})

nplv.open(fnirs)

fnirs.write_snirf("path/to/your_new_file.snirf") # WARNING : Dont overwrite data you want to keep
```
# NeuroPipeline

`NeuroPipeline` is a tool for quick and easy to use preprocessing and visualization of Functional Near-Infrared Spectroscopy (fNIRS) data.  

## Usage

```python
from neuropipeline.fnirs import fNIRS
import neuropipeline.visualizer as nplv

fnirs = fNIRS("path/to/your_file.snirf")
fnirs.preprocess(optical_density=True,
                 hemoglobin_concentration=True,
                 motion_correction=True,
                 temporal_filtering=True,
                 detrending=True,
                 normalization=False
                 )

nplv.open(fnirs)

snirf.write_snirf("path/to/your_new_file.snirf") # WARNING: Be cautious not to overwrite any data you want to keep. 
```
## Analysis Example: Heel Stimulation

These plots display data from a single subject during a robotic heel-stimulation experiment, showing the time Series, spectrogram, and frequency for two different scenarios. The vertical red dashed lines indicate block onset (rest then stimuli). In this experiment, a robot mechanically stimulated the heel 6 times. In the Supination case (left) oxygenated hemoglobin (HbO) increases as the stimuli begins. This is supported by the spectrogram, where we see increased activity at 0.025 Hz (the neurogenic band) that align well with the robot's movements. This confirms the pipeline has successfully captured brain activity in the sensory cortex. In contrast, the Pronation case (right) do not show the same clear correlation with the stimuli onset.

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/adamaske/neuropipeline/main/visualization_example.jpg" alt="Supination case"/></td>
    <td><img src="https://raw.githubusercontent.com/adamaske/neuropipeline/main/visualization_example_pronation.jpg" alt="Pronation case"/></td>
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
from neuropipeline import fNIRS, fNIRSPreprocessor

from neuropipeline.fnirs import visualizer as nplv

fnirs = fNIRS("path/to/your_file.snirf")

# Advanced Preprocessing Configuration
pp = fNIRSPreprocessor(fnirs) # Create preprocesssor
pp.set_optical_density(True) # Configure
pp.set_hemoglobin_concentration(True)
pp.set_motion_correction(True)
pp.set_temporal_filtering(True, lowcut=0.01, highcut=0.2, order=15)
pp.set_detrending(True)
pp.set_normalization(False)

pp.print() # Inspect the settings

fnirs.preprocess(pp) # Pass the preprocesser only

fnirs.write_snirf("path/to/your_new_file.snirf") # WARNING : Dont overwrite data you want to keep

nplv.set_spectrogram_limits(0.0, 0.2) # The spectrogram will show frequencies from  0 to 0.2 Hz 

nplv.set_marker_dictionary({2:"Rest",      # Display as text rather than indices
                             3:"Stimuli A", 
                             4:"Stimuli B"})

nplv.set_spectrum_mode("FFT") # What type of spectrum to show: "FFT" or "PSD"

# NOTE : The wavelet method is computationally intensive
# Try "STFT" first, then "Wavelet" if needed
nplv.set_spectrogram_method("Wavelet") 

nplv.open(fnirs)
```

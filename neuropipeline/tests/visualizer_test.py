from neuropipeline import fNIRS, fNIRSPreprocessor

from neuropipeline.fnirs import visualizer as nplvf
from neuropipeline.eeg import visualizer as nplve

fnirs = fNIRS("C:/dev/NIRS_Cardiac_Analysis/data/RH-data/Second/2026-01-14_003_HP08_TDDR_TRIM.snirf")
# More advanced preprocessing configuration 
pp = fNIRSPreprocessor(fnirs) # Create a preprocessor object
pp.set_optical_density(True)
pp.set_hemoglobin_concentration(True)
pp.set_motion_correction(True)
pp.set_temporal_filtering(True, lowcut=0.01, highcut=0.2, order=15)
pp.set_detrending(True)
pp.set_normalization(False)

pp.print() # Inspect the settings

#fnirs.preprocess(pp) # Apply preprocessing to the fNIRS object

nplvf.set_spectrogram_limits(0.0, 0.2)

nplvf.set_marker_dictionary({2:"Rest", 
                             3:"Stimuli A", 
                             4:"Stimuli B"})

nplvf.set_spectrum_mode("FFT") # What type of spectrum to show: "FFT" or "PSD"

# NOTE : The wavelet method is computationally intensive
# Try "STFT" first, then "Wavelet" if needed
nplvf.set_spectrogram_method("STFT") 

nplvf.open(fnirs) 

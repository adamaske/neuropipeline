from neuropipeline import fNIRS
from neuropipeline.fnirs import visualizer

snirf_path = "C:\\dev\\NIRWizard\\examples\\example_snirf_data\\2025-05-19_004.snirf"
f = fNIRS(snirf_path)

f.to_optical_density().tddr().to_hemoglobin().bandpass(0.02, 0.1)


f.to_snirf("c:\\dev\\NIRWizard\\examples\\example_snirf_data\\hand_processed.snirf")

visualizer.set_spectrogram_limits(0, 5)
visualizer.open(f)

from neuropipeline import fNIRS
from neuropipeline.fnirs import visualizer

snirf_path = "C:\\dev\\NIRWizard\\examples\\example_snirf_data\\2025-05-19_004.snirf"
snirf_path = "neuropipeline\\tests\\data\\2025-05-19_004_Satori12Match.snirf"
f = fNIRS(snirf_path)

visualizer.set_spectrogram_limits(0, 0.12)
visualizer.open(f)

exit()
f.to_optical_density().tddr().to_hemoglobin().bandpass(0.02, 0.1)


f.to_snirf("c:\\dev\\NIRWizard\\examples\\example_snirf_data\\hand_processed.snirf")


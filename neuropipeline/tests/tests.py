from neuropipeline.fnirs import fNIRS
from neuropipeline.eeg import EEG

eeg = EEG()
snirf = fNIRS()
snirf.read_snirf("C:/dev/neuro-glial-analysis/data/Subject01/Trial 3 - Supination/2025-03-24_003.snirf")
snirf.feature_epochs(2, 2, 5)
exit()
snirf.wl_to_od()
snirf.od_to_hb()
snirf.write_snirf("C:/dev/test_snirf.snirf")
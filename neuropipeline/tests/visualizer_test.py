from neuropipeline.fnirs import fNIRS
import neuropipeline.visualizer as nplv
import matplotlib.pyplot as plt

def run_test(snirf_filepath, preprocess):
    
    fnirs = fNIRS(snirf_filepath)
    
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
        2: "Rest",
        3: "P",
        4: "S",
    })

    nplv.open(fnirs)
    
if __name__ == "__main__":
    #run_test("C:\\nirs\\data\\2026-01-12\\relaxed_infront_pc\\2026-01-12_002.snirf", True)
    #run_test("C:\\nirs\\data\\Subject03 -Only fNIRS\\Trial 3\\2025-04-01_005.snirf", True)
    run_test("C:\\nirs\\data\\sub03_trial05.snirf", True)
    exit()
    # Supination case
    run_test("C:/dev/neuro-glial-analysis/data/Subject01/Trial 3 - Supination/2025-03-24_003.snirf",
             True)

        # Pronation case
    run_test("C:/dev/neuro-glial-analysis/data/Subject01/Trial 4 - Pronation/2025-03-24_004.snirf",
             True)

    # No preprocessing case
    run_test("C:/dev/neuro-glial-analysis/data/Subject01/Trial 3 - Supination/2025-03-24_003.snirf",
             False)
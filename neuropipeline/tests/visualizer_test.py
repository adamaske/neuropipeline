from neuropipeline.fnirs import fNIRS
import neuropipeline.visualizer as nplv

if __name__ == "__main__":

    fnirs = fNIRS("C:/dev/neuro-glial-analysis/data/Subject01/Trial 3 - Supination/2025-03-24_003.snirf")
    fnirs.preprocess(optical_density=True,
                     hemoglobin_concentration=True,
                     motion_correction=True,
                     temporal_filtering=True,
                     detrending=True,
                     normalization=False
                     )
    
    #fnirs.write_snirf("C:/dev/neuro-glial-analysis/data/Subject01/Trial 3 - Supination/2025-03-24_003_processed.snirf")
    nplv.set_spectrogram_limits(0.0, 2)
    nplv.open(fnirs)
    
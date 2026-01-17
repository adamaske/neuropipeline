import neuropipeline as npl

from neuropipeline.fnirs import fNIRS
import neuropipeline.visualizer as nplv
import numpy as np


if __name__ == "__main__":
    
    fnirs = npl.fnirs.fNIRS("C:/dev/neuro-glial-analysis/data/Subject01/Trial 3 - Supination/2025-03-24_003.snirf")
    fnirs.preprocess(optical_density=True, 
                     hemoglobin_concentration=True, 
                     temporal_filtering=True,
                     normalization=False, 
                     detrending=True)
    nplv.open(fnirs)
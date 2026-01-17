import neuropipeline as npl

from neuropipeline.fnirs import fNIRS
import neuropipeline.visualizer as nplv

if __name__ == "__main__":

    fnirs = fNIRS("C:\\nirs\\Datasets\\ms_data\\Subject03\\sub03_trial03.snirf")
    fnirs.preprocess(
        optical_density=True,
        hemoglobin_concentration=True,
        temporal_filtering=True,
        normalization=False,
        detrending=True
    )
    
    nplv.open(fnirs)
    
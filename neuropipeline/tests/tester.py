from neuropipeline import fNIRS
import neuropipeline.fnirs.visualizer as nplv

fnirs = fNIRS("path/to/your/datafile.snirf")
fnirs.preprocess(optical_density=True,
                 hemoglobin_concentration=True,
                 motion_correction=True,
                 temporal_filtering=True,
                 detrending=True,
                 normalization=False # Only use normalization if necessary
                 )

# WARNING: Be cautious not to overwrite any data you want to keep.
fnirs.write_snirf("path/to/your/processed_file.snirf") 

nplv.open(fnirs)
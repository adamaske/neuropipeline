from base import Base
import numpy as np
import matplotlib.pyplot as plt


from enum import Enum

from mne.io import read_raw_snirf
from mne_nirs.io import write_raw_snirf
from snirf import validateSnirf
from mne.preprocessing.nirs import optical_density, beer_lambert_law,  temporal_derivative_distribution_repair

class fnirs_data_type(Enum):
    Wavelength = "Wavelength"
    OpticalDensity = "Optical Density"
    HemoglobinConcentration = "Hemoglobin Concentration"

WL = fnirs_data_type.Wavelength
OD = fnirs_data_type.OpticalDensity
CC = fnirs_data_type.HemoglobinConcentration

class fNIRS(Base):
    def __init__(self): 
        self.type = WL
        self.snirf = None 
        
        self.sampling_frequency = None
        self.channel_names = None
        self.channel_data = None
        self.channel_num = None
        
        self.feature_onsets = None
        self.feature_descriptions = None
    
    def print(self):
        print("sampling_frequency : ", self.sampling_frequency, " Hz")
        print("channel_num : ", self.channel_num)
        print("channel_data : ", self.channel_data.shape)
        print("channel_names : ", self.channel_names)
        print("feature_onsets : ", self.feature_onsets)
        print("feature_descriptions : ", self.feature_descriptions)
        
    def read_snirf(self, filepath):
        print(f"Reading SNIRF from {filepath}")
        result = validateSnirf(filepath)
        print("valid : ", result.is_valid())
        self.snirf = read_raw_snirf(filepath)
        
        # fNIRS info
        info = self.snirf.info
        self.sampling_frequency = float(info["sfreq"])
        self.channel_names = info["ch_names"]
        self.channel_data = np.array(self.snirf.get_data())
        self.channel_num = int(info["nchan"])
        # Features
        annotations = self.snirf._annotations
        self.feature_onsets = np.array(annotations.onset, dtype=float)
        self.feature_descriptions = np.array(annotations.description, dtype=int)
        
        self.channel_dict = {}
        for i, channel_name in enumerate(self.channel_names):
            
            source_detector = channel_name.split()[0]
            wavelength = channel_name.split()[1]
            
            if source_detector not in self.channel_dict:
                self.channel_dict[source_detector] = {"HbO" : None, 
                                                 "HbR" : None
                                                 }
            
            channel_data = self.channel_data[i] 
            
            if wavelength == "HbR".lower() or wavelength == "760":
                self.channel_dict[source_detector]["HbR"] = channel_data
                
            if wavelength == "HbO".lower() or wavelength == "850":
                self.channel_dict[source_detector]["HbO"] = channel_data
                
        
    def write_snirf(self, filepath):
        write_raw_snirf(self.snirf, filepath)
        print(f"Wrote SNIRF to {filepath}")
        result = validateSnirf(filepath)
        print("valid : ", result.is_valid())
        
        
    def wl_to_od(self):
        if self.type != WL:
            print(f"sNIRF type is {self.type}, cannot convert to {OD}!")
            return
        self.snirf = optical_density(self.snirf)
        self.type = OD

    def od_to_hb(self):
        if self.type != OD:
            print(f"sNIRF type is {self.type}, cannot convert to {CC}!")
            return 
        self.snirf = beer_lambert_law(self.snirf)
        self.type = CC

    def feature_epochs(self, feature_description, tmin, tmax):
        
        onsets = [] # Fill with the onsets
        print(self.feature_descriptions)
        print(self.feature_onsets)
        for i, desc in enumerate(self.feature_descriptions):
            if desc == feature_description:
                onsets.append(self.feature_onsets[i])
        
        print("onsets : ", onsets)
        
        exit()
        for i, channel_name in enumerate(self.channel_dict):
            
            pass
        
        
        pass
snirf = fNIRS()
snirf.read_snirf("C:/dev/neuro-glial-analysis/data/Subject01/Trial 3 - Supination/2025-03-24_003.snirf")
snirf.feature_epochs(2, 2, 5)
exit()
snirf.wl_to_od()
snirf.od_to_hb()
snirf.write_snirf("C:/dev/test_snirf.snirf")
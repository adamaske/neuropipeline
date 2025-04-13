import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from mne.io import read_raw_snirf
from mne_nirs.io import write_raw_snirf
from snirf import validateSnirf
from mne.preprocessing.nirs import optical_density, beer_lambert_law,  temporal_derivative_distribution_repair
from mne import Annotations
from scipy.signal import butter, sosfreqz, sosfiltfilt, iirnotch, filtfilt, welch
from scipy.stats import pearsonr

def butter_bandpass(lowcut, highcut, fs, freqs=512, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    w, h = sosfreqz(sos, worN=None, whole=True, fs=fs)
    return sos, w, h

def butter_bandpass_filter(time_series, lowcut, highcut, fs, order):
    sos, w, h = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, time_series)
    return np.array(y)

def notch_filter(data, sfreq, freqs=[50, 60]):
    """Apply notch filters at specified frequencies."""
    for freq in freqs:
        b, a = iirnotch(freq, 30, sfreq)
        data = filtfilt(b, a, data, axis=-1)
    return data

from scipy.signal import freqz, sosfreqz

def compute_fft(time_series, fs, freq_limit:float|None):
    # Compute FFT
    N = len(time_series)  # Length of the signal
    fft_result = np.fft.fft(time_series)
    fft_freq = np.fft.fftfreq(N, d=1/fs)#/fs)  # Frequency axis

    # Take the positive half of the spectrum
    positive_freqs = fft_freq[:N // 2]
    positive_spectrum = np.abs(fft_result[:N // 2]) * (2 / N)  # Normalize for one-sided
    
    if freq_limit is None:
        return positive_freqs, positive_spectrum

    # Filter frequencies to only include up to freq_limit
    indices = positive_freqs <= freq_limit
    limited_freqs = positive_freqs[indices]
    limited_spectrum = positive_spectrum[indices]
    return limited_freqs, limited_spectrum

def compute_psd(time_series, fs, freq_limit:float|None):
    # Compute FFT
    freqs, spectrum = compute_fft(time_series, fs, freq_limit)
    # Normalize to get power spectral density
    psd = np.square(spectrum) / (fs * len(time_series))  
    # Double the PSD for one-sided spectrum (except at DC and Nyquist)
    psd[1:] = 2 * psd[1:]
    return freqs, psd

class fnirs_data_type(Enum):
    Wavelength = "Wavelength"
    OpticalDensity = "Optical Density"
    HemoglobinConcentration = "Hemoglobin Concentration"

WL = fnirs_data_type.Wavelength
OD = fnirs_data_type.OpticalDensity
CC = fnirs_data_type.HemoglobinConcentration

class fNIRS():
    def __init__(self, filepath=None): 
        self.type = WL
        self.snirf = None 
        
        self.sampling_frequency = None
        self.channel_names = None
        self.channel_data = None
        self.channel_num = None
        
        self.feature_onsets = None
        self.feature_descriptions = None
        
        if filepath != None:
            self.read_snirf(filepath)
    
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
        snirf = read_raw_snirf(filepath)
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
        self.feature_durations = np.array(annotations.duration, dtype=float)

        self.channel_dict = None
        
    def update_snirf_object(self):
        
        # Overwrite channel data
        self.snirf._data = self.channel_data
        
        # Fix Annotations
        new_annotations = Annotations(onset=self.feature_onsets,
                                      duration=self.feature_durations,
                                      description=self.feature_descriptions)
        self.snirf.set_annotations(new_annotations)
        
    def get_channel_dict(self):
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
        return self.channel_dict
    
    def split(self):
        """
        Splits the channel data into HbO and HbR
        """
        assert(len(self.channel_data) == len(self.channel_names))
        
        hbo_channels = []
        hbo_names = []
        hbr_channels = []
        hbr_names = []
        for i, channel_name in enumerate(self.channel_names): 
            parts = channel_name.split()
            
            assert(len(parts) == 2)
            
            source_detector, wavelength = parts[0], parts[1].lower()
        
            if wavelength == "hbr" or wavelength == "760":
                hbr_channels.append(self.channel_data[i] )
                hbr_names.append(source_detector)
                
            if wavelength == "hbo" or wavelength == "850":
                hbo_channels.append(self.channel_data[i] )
                hbo_names.append(source_detector)
        
        # Into numpy arrays!
        hbo_channels = np.array(hbo_channels)
        hbr_channels = np.array(hbr_channels)
        
        return hbo_channels, hbo_names, hbr_channels, hbr_names
      
    def write_snirf(self, filepath):
        write_raw_snirf(self.snirf, filepath)
        print(f"Wrote SNIRF to {filepath}")
        result = validateSnirf(filepath)
        print("valid : ", result.is_valid())
        

    def to_optical_density(self, use_inital_value=False):
        """
        Converts raw light intensity data to optical density. \nif use_inital_value is False then this function mimicks MNE's optical_density function.

        Parameters:
            raw_data (numpy.ndarray): 2D array [channels x samples] of raw light intensities.

        Returns:
            optical_density (numpy.ndarray): Converted optical density data.
        """
        
        # We need to compare how different this result is from the mne ones is
        
        if self.type != WL:
            print(f"sNIRF type is {self.type}, cannot convert to {OD}!")
            return
        
        if use_inital_value: # Use I_0 according to Dans 2019
            measured_intensity = self.channel_data
            initial_intensity = measured_intensity[:, 0]

            # Avoid division by zero
            safe_intensity = np.clip(measured_intensity, a_min=1e-12, a_max=None)
            safe_initial = np.clip(initial_intensity[:, np.newaxis], a_min=1e-12, a_max=None)

            od = -np.log(safe_intensity / safe_initial)
        else: # Use mean according to MNE.nirs
            
            data = np.abs(self.channel_data)  # Take absolute to avoid negative intensities

            # Replace zeros by the smallest positive value per channel
            min_nonzero = np.min(np.where(data > 0, data, np.inf), axis=1, keepdims=True)
            data = np.maximum(data, min_nonzero)

            # Normalize each channel by its mean
            means = np.mean(data, axis=1, keepdims=True)
            normalized = data / means

            # Apply natural log and invert sign
            od = -np.log(normalized)
        
        self.channel_data = od
        self.type = OD
        ch_dict = {}
        for ch_name in self.channel_names:
            ch_dict[ch_name] = "fnirs_od"
            
        self.snirf.set_channel_types(mapping=ch_dict)

    def to_hemoglobin_concentration(self):
        if self.type != OD:
            print(f"sNIRF type is {self.type}, cannot convert to {CC}!")
            return 
        
        self.update_snirf_object()
        hb = beer_lambert_law(self.snirf)
        
        self.snirf = hb
        
        # TODO : Do we actually need to reread all this?
        info = self.snirf.info
        self.channel_names = info["ch_names"]
        self.channel_data = np.array(self.snirf.get_data())
    

    def feature_epochs(self, feature_description, tmin, tmax):
        
        onsets = [] # Fill with the onsets
        print(self.feature_descriptions)
        print(self.feature_onsets)
        for i, desc in enumerate(self.feature_descriptions):
            if desc == feature_description:
                onsets.append(self.feature_onsets[i])
        print("feature : ", feature_description, f" ({len(onsets)})")
        print("onsets : ", onsets)
        
        exit()
        for i, channel_name in enumerate(self.channel_dict):
            
            pass
        
        
        pass
    
    
    def remove_features(self, features_to_remove):
        """Removes features by description match."""
    
        indices_to_keep = [
            i for i, desc in enumerate(self.feature_descriptions) 
            if desc not in features_to_remove
        ]

        self.feature_onsets = np.array([self.feature_onsets[i] for i in indices_to_keep])
        self.feature_descriptions = np.array([self.feature_descriptions[i] for i in indices_to_keep])
        self.feature_durations = np.array([self.feature_durations[i] for i in indices_to_keep])

        print(f"Removed {len(self.feature_descriptions) - len(indices_to_keep)} features. Remaining: {len(self.feature_descriptions)}.")
        self.update_snirf_object()
        
    def trim_from_features(self, cut_from_first_feature:float=5, cut_from_last_feature:float=10) -> None:
        
        first_seconds = self.feature_onsets[0]
        last_seconds = self.feature_onsets[-1]
        
        first_frame = first_seconds * self.sampling_frequency
        last_frame = last_seconds * self.sampling_frequency
        
        start_seconds = first_seconds - cut_from_first_feature
        end_seconds = last_seconds + cut_from_last_feature
        
        start_frames = int(first_frame - (cut_from_first_feature * self.sampling_frequency))
        end_frames = int(last_frame + (cut_from_last_feature * self.sampling_frequency))
        
        assert(start_frames >= 0)
        assert(end_frames < self.channel_data.shape[1])
        print(f"Trimming From Features : [ {start_seconds} : {end_seconds} ] / [ {start_frames} : {end_frames} ]")
        
        self.channel_data = self.channel_data[:, start_frames:end_frames]
        
        valid_indices = [i for i, onset in enumerate(self.feature_onsets) if start_seconds <= onset < end_seconds]
        self.feature_onsets = np.array([self.feature_onsets[i] - start_seconds for i in valid_indices])
        self.feature_descriptions = np.array([self.feature_descriptions[i] for i in valid_indices])
        self.feature_durations = np.array([self.feature_durations[i] for i in valid_indices])
        # OVERWRITE THE .SNIRF
        self.update_snirf_object()
    
    def bandpass_channels(self, low_freq=0.01, high_freq=0.1, order=5):
        """
        Applies a digital bandpass filter to all channels. Returns filtered snirf object. 

        Args:
            snirf (RawSNIRF) : RawSNIRF object
            l_freq : Lowcut frequency, the lower edge of passband
            h_freq : Highcut frequency, the high edge of passband  
            n : Filter order, higher means small transition band

        Returns:
            filtered (RawSNIRF) : New RawSNIRF object with filtered channels
        """
        for i, channel in enumerate(self.channel_data):  # Iterate over each channel
            self.channel_data[i] = butter_bandpass_filter(channel, low_freq, high_freq, self.sampling_frequency, order)
        
    def preprocess(self, optical_density=True, hemoglobin_concentration=True, temporal_filtering=True, normalization=True):
        if optical_density:
            self.to_optical_density(use_inital_value=False)

        if hemoglobin_concentration:
            self.to_hemoglobin_concentration()
    
        if temporal_filtering:
            lowcut = 0.02
            highcut = 0.07
            order = 15
            self.bandpass_channels(lowcut, highcut, order)
        
        if normalization:
            mean_vals = np.mean(self.channel_data, axis=1, keepdims=True)  # Compute mean per channel
            std_vals = np.std(self.channel_data, axis=1, keepdims=True)  # Compute standard deviation per channel

            std_vals[std_vals == 0] = 1

            normalized_channels = (self.channel_data - mean_vals) / std_vals  # Apply Z-Normalization

            self.channel_data = normalized_channels
            self.snirf._data = normalized_channels
                
    def plot_channels(self,):
        
        hbo_data, hbo_names, hbr_data, hbr_names = self.split()
    
        plt.figure(figsize=(12, 8))

        # Plot HbO time series
        plt.subplot(2, 2, 1)
        for i, ch in enumerate(hbo_data):
            plt.plot(ch, label=f"{hbo_names[i]}")
        plt.title("HbO Time Series")
        plt.legend()

        # Plot HbR time series
        plt.subplot(2, 2, 2)
        for i, ch in enumerate(hbr_data):
            plt.plot(ch, label=f"{hbr_names[i]}")
        plt.title("HbR Time Series")
        plt.legend()

        # Plot HbO Power Spectral Density
        plt.subplot(2, 2, 3)
        for i, ch in enumerate(hbo_data):
            freqs, spectra = compute_psd(ch, self.sampling_frequency, int(self.sampling_frequency/2))
            plt.plot(freqs, spectra)
        plt.title("HbO : Power Spectral Density")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD [V²/Hz]")
        plt.legend()

        # Plot HbR Power Spectral Density
        plt.subplot(2, 2, 4)
        for i, ch in enumerate(hbr_data):
            freqs, spectra = compute_psd(ch, self.sampling_frequency, int(self.sampling_frequency/2))
            plt.plot(freqs, spectra)
        plt.title("HbR : Power Spectral Density")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD [V²/Hz]")
        plt.legend()

        plt.tight_layout()
        plt.show()
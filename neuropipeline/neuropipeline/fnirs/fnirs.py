import numpy as np
import matplotlib.pyplot as plt
import h5py
from enum import Enum
from mne.io import read_raw_snirf
from mne_nirs.io import write_raw_snirf
from snirf import validateSnirf
from mne.preprocessing.nirs import beer_lambert_law
from mne import Annotations
from scipy.signal import detrend

from .preprocessor import (
    TDDR,
    butter_bandpass_filter,
    fNIRSPreprocessor,
)

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

    def get_duration(self) -> float:
        """
        Returns the total duration of the recording in seconds.

        Returns:
            float: Duration in seconds.
        """
        n_samples = self.channel_data.shape[1]
        return n_samples / self.sampling_frequency

    def get_time(self) -> np.ndarray:
        """
        Returns a time vector for the recording.

        Returns:
            np.ndarray: Array of time points in seconds, starting from 0.
        """
        n_samples = self.channel_data.shape[1]
        return np.arange(n_samples) / self.sampling_frequency

    def read_snirf(self, filepath):
        print(f"Reading SNIRF from {filepath}")
        self._filepath = filepath
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

    def get_metadata(self, filepath: str = None) -> dict:
        """
        Extracts metaDataTags from a SNIRF file using h5py.

        Args:
            filepath: Path to the SNIRF file. If None, uses the filepath from
                      the last read_snirf call (stored in self._filepath).

        Returns:
            dict: Dictionary containing all metadata tags from the SNIRF file.
                  Keys are the tag names, values are the decoded string values.
        """
        if filepath is None:
            filepath = getattr(self, '_filepath', None)
            if filepath is None:
                raise ValueError("No filepath provided and no file has been read yet.")

        metadata = {}

        with h5py.File(filepath, 'r') as f:
            # SNIRF files have /nirs or /nirs1, /nirs2, etc. for multiple recordings
            # Check for metaDataTags in the first nirs group
            nirs_groups = [key for key in f.keys() if key.startswith('nirs')]

            if not nirs_groups:
                print("Warning: No 'nirs' group found in SNIRF file.")
                return metadata

            # Use the first nirs group (or 'nirs' if it exists)
            nirs_key = 'nirs' if 'nirs' in nirs_groups else sorted(nirs_groups)[0]
            nirs_group = f[nirs_key]

            if 'metaDataTags' not in nirs_group:
                print("Warning: No 'metaDataTags' found in SNIRF file.")
                return metadata

            meta_group = nirs_group['metaDataTags']

            for tag_name in meta_group.keys():
                tag_data = meta_group[tag_name]

                # Handle different data types
                if isinstance(tag_data, h5py.Dataset):
                    value = tag_data[()]

                    # Decode bytes to string if needed
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    elif isinstance(value, np.ndarray):
                        # Handle array of bytes (common in SNIRF)
                        if value.dtype.kind == 'S' or value.dtype.kind == 'O':
                            if value.ndim == 0:
                                value = value.item()
                                if isinstance(value, bytes):
                                    value = value.decode('utf-8')
                            else:
                                value = [v.decode('utf-8') if isinstance(v, bytes) else v for v in value.flat]
                                if len(value) == 1:
                                    value = value[0]

                    metadata[tag_name] = value

        return metadata

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

        return hbo_channels, hbo_names, hbr_channels, hbr_names
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
        # Ensure the snirf object is up to date with our local data
        self.update_snirf_object()

        try:
            write_raw_snirf(self.snirf, filepath)
            print(f"Wrote SNIRF to {filepath}")
            result = validateSnirf(filepath)
            print("valid : ", result.is_valid())
        except (KeyError, ValueError) as e:
            # mne_nirs may fail with hemoglobin concentration data
            # Fall back to writing with h5py directly
            print(f"mne_nirs write failed ({e}), using h5py fallback...")
            self._write_snirf_h5py(filepath)

    def _write_snirf_h5py(self, filepath):
        """
        Write SNIRF file using h5py directly.
        Fallback when mne_nirs fails (e.g., with hemoglobin concentration data).
        """
        import os

        # Read the original file to preserve structure
        if not hasattr(self, '_filepath') or self._filepath is None:
            raise ValueError("No original file to base structure on. Cannot use h5py fallback.")

        # Copy original and modify
        import shutil
        shutil.copy(self._filepath, filepath)

        with h5py.File(filepath, 'r+') as f:
            # Find the nirs group
            nirs_groups = [key for key in f.keys() if key.startswith('nirs')]
            nirs_key = 'nirs' if 'nirs' in nirs_groups else sorted(nirs_groups)[0]
            nirs_group = f[nirs_key]

            # Update the data block
            if 'data1' in nirs_group:
                data_group = nirs_group['data1']

                # Update dataTimeSeries
                if 'dataTimeSeries' in data_group:
                    del data_group['dataTimeSeries']
                data_group.create_dataset('dataTimeSeries', data=self.channel_data.T)

                # Update time vector
                if 'time' in data_group:
                    del data_group['time']
                time_vector = np.arange(self.channel_data.shape[1]) / self.sampling_frequency
                data_group.create_dataset('time', data=time_vector)

            # Update stim blocks (markers/features)
            # First, remove existing stim blocks
            stim_keys = [k for k in nirs_group.keys() if k.startswith('stim')]
            for key in stim_keys:
                del nirs_group[key]

            # Create new stim blocks for each unique marker type
            if len(self.feature_onsets) > 0:
                unique_descriptions = sorted(set(self.feature_descriptions))

                for i, desc in enumerate(unique_descriptions, start=1):
                    stim_group = nirs_group.create_group(f'stim{i}')

                    # Get indices for this description
                    mask = self.feature_descriptions == desc
                    onsets = self.feature_onsets[mask]
                    durations = self.feature_durations[mask]
                    amplitudes = np.ones(len(onsets))

                    # Create data matrix [onset, duration, amplitude]
                    stim_data = np.column_stack([onsets, durations, amplitudes])
                    stim_group.create_dataset('data', data=stim_data)

                    # Store name
                    stim_group.create_dataset('name', data=str(desc).encode('utf-8'))

        print(f"Wrote SNIRF to {filepath} (h5py fallback)")
        result = validateSnirf(filepath)
        print("valid : ", result.is_valid())

    def downsample(self, factor:int):
        """
        Downsamples the fNIRS data by an integer factor.

        Args:
            factor: Integer downsampling factor.
        """
        if factor <= 1:
            print("Downsample factor must be greater than 1.")
            return
        
        # Downsample channel data
        self.channel_data = self.channel_data[:, ::factor]
        
        # Update sampling frequency
        self.sampling_frequency /= factor
        
        # Update the snirf object
        self.update_snirf_object()
        print(f"Downsampled by a factor of {factor}. New sampling frequency: {self.sampling_frequency} Hz")

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

    def add_features(self,
                     onsets: np.ndarray | list,
                     descriptions: np.ndarray | list,
                     durations: np.ndarray | list | float = 0.0,
                     sort: bool = True) -> None:
        """
        Adds new feature markers to the existing data.

        Args:
            onsets: Array of onset times in seconds for the new features.
            descriptions: Array of feature descriptions/labels for the new features.
            durations: Array of durations in seconds, or a single value applied to all.
                       Defaults to 0.0.
            sort: If True, sorts all features by onset time after adding. Default True.

        Raises:
            ValueError: If onsets and descriptions have different lengths.

        Usage:
            # Add multiple features
            fnirs.add_features(
                onsets=[10.0, 20.0, 30.0],
                descriptions=[1, 2, 1],
                durations=[5.0, 5.0, 5.0]
            )

            # Add with uniform duration
            fnirs.add_features(
                onsets=[10.0, 20.0],
                descriptions=[3, 3],
                durations=2.0
            )

            # Add a single feature
            fnirs.add_features(onsets=[15.0], descriptions=[4], durations=1.0)
        """
        onsets = np.array(onsets, dtype=float)
        descriptions = np.array(descriptions)

        if len(onsets) != len(descriptions):
            raise ValueError(f"onsets length ({len(onsets)}) must match descriptions length ({len(descriptions)})")

        # Handle durations
        if np.isscalar(durations):
            durations = np.full(len(onsets), float(durations))
        else:
            durations = np.array(durations, dtype=float)
            if len(durations) != len(onsets):
                raise ValueError(f"durations length ({len(durations)}) must match onsets length ({len(onsets)})")

        # Append to existing features
        self.feature_onsets = np.concatenate([self.feature_onsets, onsets])
        self.feature_descriptions = np.concatenate([self.feature_descriptions, descriptions])
        self.feature_durations = np.concatenate([self.feature_durations, durations])

        # Sort by onset time if requested
        if sort:
            sort_indices = np.argsort(self.feature_onsets)
            self.feature_onsets = self.feature_onsets[sort_indices]
            self.feature_descriptions = self.feature_descriptions[sort_indices]
            self.feature_durations = self.feature_durations[sort_indices]

        print(f"Added {len(onsets)} features. Total: {len(self.feature_onsets)}.")
        self.update_snirf_object()

    def replace_features(self,
                         onsets: np.ndarray | list = None,
                         descriptions: np.ndarray | list = None,
                         durations: np.ndarray | list = None) -> None:
        """
        Replaces the feature markers in the data.

        Args:
            onsets: Array of onset times in seconds. If None, keeps existing onsets.
            descriptions: Array of feature descriptions/labels. If None, keeps existing.
            durations: Array of durations in seconds. If None, keeps existing durations.
                       If a single value is provided, it will be applied to all features.

        Raises:
            ValueError: If array lengths don't match when multiple arrays are provided.

        Usage:
            # Replace all features completely
            fnirs.replace_features(
                onsets=[10.0, 20.0, 30.0],
                descriptions=[1, 2, 1],
                durations=[5.0, 5.0, 5.0]
            )

            # Replace only onsets, keep existing descriptions and durations
            fnirs.replace_features(onsets=[10.0, 20.0, 30.0])

            # Use uniform duration for all features
            fnirs.replace_features(
                onsets=[10.0, 20.0, 30.0],
                descriptions=[1, 2, 1],
                durations=5.0
            )
        """
        # Determine the reference length
        if onsets is not None:
            onsets = np.array(onsets, dtype=float)
            ref_length = len(onsets)
        elif descriptions is not None:
            descriptions = np.array(descriptions)
            ref_length = len(descriptions)
        elif durations is not None and not np.isscalar(durations):
            durations = np.array(durations, dtype=float)
            ref_length = len(durations)
        else:
            ref_length = len(self.feature_onsets)

        # Process onsets
        if onsets is not None:
            if len(onsets) != ref_length:
                raise ValueError(f"onsets length ({len(onsets)}) doesn't match expected length ({ref_length})")
            self.feature_onsets = onsets
        elif ref_length != len(self.feature_onsets):
            raise ValueError(f"Cannot keep existing onsets: length mismatch ({len(self.feature_onsets)} vs {ref_length})")

        # Process descriptions
        if descriptions is not None:
            descriptions = np.array(descriptions)
            if len(descriptions) != ref_length:
                raise ValueError(f"descriptions length ({len(descriptions)}) doesn't match expected length ({ref_length})")
            self.feature_descriptions = descriptions
        elif ref_length != len(self.feature_descriptions):
            raise ValueError(f"Cannot keep existing descriptions: length mismatch ({len(self.feature_descriptions)} vs {ref_length})")

        # Process durations
        if durations is not None:
            if np.isscalar(durations):
                # Single value: apply to all features
                self.feature_durations = np.full(ref_length, float(durations))
            else:
                durations = np.array(durations, dtype=float)
                if len(durations) != ref_length:
                    raise ValueError(f"durations length ({len(durations)}) doesn't match expected length ({ref_length})")
                self.feature_durations = durations
        elif ref_length != len(self.feature_durations):
            raise ValueError(f"Cannot keep existing durations: length mismatch ({len(self.feature_durations)} vs {ref_length})")

        print(f"Replaced features: {ref_length} markers set.")
        self.update_snirf_object()

    def trim(self, start_seconds: float = 0, end_seconds: float = 0) -> None:
        """
        Trims the recording by cutting X seconds from the start and Y seconds from the end.

        Args:
            start_seconds: Number of seconds to cut from the beginning of the recording.
            end_seconds: Number of seconds to cut from the end of the recording.
        """
        total_samples = self.channel_data.shape[1]
        total_duration = total_samples / self.sampling_frequency

        if start_seconds + end_seconds >= total_duration:
            raise ValueError(f"Cannot trim {start_seconds}s from start and {end_seconds}s from end. "
                             f"Total duration is only {total_duration:.2f}s.")

        tmin = start_seconds
        tmax = total_duration - end_seconds

        print(f"Trimming: removing {start_seconds}s from start, {end_seconds}s from end")
        print(f"  Before: {total_samples} samples ({total_duration:.2f}s)")

        # Use MNE's crop to properly trim the Raw object
        self.snirf.crop(tmin=tmin, tmax=tmax)

        # Update our local data to match
        self.channel_data = np.array(self.snirf.get_data())

        new_duration = self.channel_data.shape[1] / self.sampling_frequency
        print(f"  After: {self.channel_data.shape[1]} samples ({new_duration:.2f}s)")

        # Update feature arrays from the cropped annotations
        annotations = self.snirf._annotations
        self.feature_onsets = np.array(annotations.onset, dtype=float)
        self.feature_descriptions = np.array(annotations.description, dtype=int)
        self.feature_durations = np.array(annotations.duration, dtype=float)

    def trim_from_features(self, cut_from_first_feature:float=5, cut_from_last_feature:float=10) -> None:
        
        first_seconds = self.feature_onsets[0]
        last_seconds = self.feature_onsets[-1]
        
        first_frame = first_seconds * self.sampling_frequency
        last_frame = last_seconds * self.sampling_frequency
        
        start_seconds = first_seconds - cut_from_first_feature
        end_seconds = last_seconds + cut_from_last_feature
        
        start_frames = int(first_frame - (cut_from_first_feature * self.sampling_frequency))
        end_frames = int(last_frame + (cut_from_last_feature * self.sampling_frequency))
        
        print(f"Trimming From Features {self.channel_data.shape} : [ {start_seconds} : {end_seconds} ] / [ {start_frames} : {end_frames} ]")
        assert(start_frames >= 0)
        if end_frames < self.channel_data.shape[1]:
            end_frames = self.channel_data.shape[1]-1
            
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
        
    def preprocess(self,
                   preprocessor: fNIRSPreprocessor = None,
                   optical_density: bool = True,
                   hemoglobin_concentration: bool = True,
                   motion_correction: bool = True,
                   temporal_filtering: bool = True,
                   detrending: bool = True,
                   normalization: bool = True):
        """
        Apply preprocessing pipeline to fNIRS data.

        Can be called with a fNIRSPreprocessor object for full control over settings,
        or with individual boolean flags for simple on/off control.

        Args:
            preprocessor: Optional fNIRSPreprocessor with configured settings.
                          If provided, other arguments are ignored.
            optical_density: Convert to optical density.
            hemoglobin_concentration: Convert to hemoglobin concentration.
            motion_correction: Apply TDDR motion correction.
            temporal_filtering: Apply bandpass filter.
            detrending: Apply linear detrending.
            normalization: Apply z-score normalization.

        Usage:
            # Simple usage with defaults
            fnirs.preprocess()

            # Toggle specific steps
            fnirs.preprocess(normalization=False)

            # Full control with preprocessor object
            pp = fNIRSPreprocessor()
            pp.set_bandpass(0.01, 0.2, order=10)
            fnirs.preprocess(pp)
        """
        # Use preprocessor settings if provided, otherwise create one from kwargs
        if preprocessor is not None:
            pp = preprocessor
        else:
            pp = fNIRSPreprocessor(
                optical_density=optical_density,
                hemoglobin_concentration=hemoglobin_concentration,
                motion_correction=motion_correction,
                temporal_filtering=temporal_filtering,
                detrending=detrending,
                normalization=normalization
            )

        if pp.optical_density:
            self.to_optical_density(use_inital_value=pp.od_use_initial_value)

        if pp.motion_correction:
            for i, channel in enumerate(self.channel_data):
                self.channel_data[i] = TDDR(channel, self.sampling_frequency)
            self.snirf._data = self.channel_data

        if pp.hemoglobin_concentration:
            self.to_hemoglobin_concentration()

        if pp.temporal_filtering:
            self.bandpass_channels(pp.bandpass_lowcut, pp.bandpass_highcut, pp.bandpass_order)

        if pp.detrending:
            for i, channel in enumerate(self.channel_data):
                self.channel_data[i] = detrend(channel, type=pp.detrend_type)
            self.snirf._data = self.channel_data

        if pp.normalization:
            mean_vals = np.mean(self.channel_data, axis=1, keepdims=True)
            std_vals = np.std(self.channel_data, axis=1, keepdims=True)
            std_vals[std_vals == 0] = 1
            normalized_channels = (self.channel_data - mean_vals) / std_vals
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
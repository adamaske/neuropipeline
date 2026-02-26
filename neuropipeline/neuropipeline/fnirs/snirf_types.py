"""
Python mirror of the C++ SNIRF data structures defined in snirf-cpp.

Struct correspondence:
    Vec2               ↔  snirfcpp/detail/Vec.h       Vec2
    Vec3               ↔  snirfcpp/detail/Vec.h       Vec3
    OptodeType         ↔  snirfcpp/detail/Probe.h     OptodeType
    Optode             ↔  snirfcpp/detail/Probe.h     Optode
    Channel            ↔  snirfcpp/detail/Probe.h     Channel
    Probe              ↔  snirfcpp/detail/Probe.h     Probe
    MetadataTag        ↔  snirfcpp/detail/Metadata.h  MetadataTag
    Metadata           ↔  snirfcpp/detail/Metadata.h  Metadata
    TimeData           ↔  snirfcpp/detail/Time.h      TimeData
    ChannelDataStore   ↔  snirfcpp/detail/Channels.h  ChannelDataStore
    EventMarker        ↔  snirfcpp/detail/Events.h    EventMarker
    Event              ↔  snirfcpp/detail/Events.h    Event
    EventsContainer    ↔  snirfcpp/detail/Events.h    EventsContainer
    AuxiliaryType      ↔  snirfcpp/detail/Biosignals.h AuxlaryType
    AuxiliaryData      ↔  snirfcpp/detail/Biosignals.h AuxilaryData
    BiosignalData      ↔  snirfcpp/detail/Biosignals.h BiosignalData
    SNIRF              ↔  snirfcpp/SNIRF.h             SNIRF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ============================================================================
# Geometry
# ============================================================================

@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0


@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


# ============================================================================
# Probe
# ============================================================================

class OptodeType(Enum):
    SOURCE   = 0
    DETECTOR = 1


@dataclass
class Optode:
    type:        OptodeType
    id:          int          # 1-indexed, mirrors C++ OptodeID
    position_2D: Vec2 | None = None
    position_3D: Vec3 | None = None


@dataclass
class Channel:
    """
    Source-detector pair with paired wavelength data.

    hbr_data  — time-series at the lower wavelength (HbR-dominant, e.g. 760 nm)
    hbo_data  — time-series at the higher wavelength (HbO-dominant, e.g. 850 nm)
    hbt_data  — element-wise sum: hbo_data + hbr_data

    Populated by ParseData1: first N/2 rows of dataTimeSeries → hbr,
    second N/2 rows → hbo, mirroring the C++ ParseData1 logic.
    """
    id:          int          # 0-indexed, mirrors C++ ChannelID
    source_id:   int          # 1-indexed
    detector_id: int          # 1-indexed
    hbo_data:    np.ndarray = field(default_factory=lambda: np.array([]))
    hbr_data:    np.ndarray = field(default_factory=lambda: np.array([]))
    hbt_data:    np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class Probe:
    channels:    dict[int, Channel]  # channel_id (0-indexed) → Channel
    sources:     dict[int, Optode]   # source_id  (1-indexed) → Optode
    detectors:   dict[int, Optode]   # detector_id(1-indexed) → Optode
    wavelengths: list[int]           # sorted wavelength values in nm


# ============================================================================
# Metadata
# ============================================================================

@dataclass
class MetadataTag:
    name:  str
    value: str


@dataclass
class Metadata:
    tags:                 list[MetadataTag] = field(default_factory=list)
    has_wings_generation: bool = False


# ============================================================================
# Time
# ============================================================================

@dataclass
class TimeData:
    time:               np.ndarray = field(default_factory=lambda: np.array([]))
    duration:           float = 0.0
    sampling_frequency: float = 0.0


# ============================================================================
# Channel data store
# ============================================================================

@dataclass
class ChannelDataStore:
    """
    Raw time-series matrix parsed from data1/dataTimeSeries.
    Layout: (channels × timepoints), mirroring C++ ChannelDataStore.
    """
    data: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    @property
    def rows(self) -> int:
        """Number of channels."""
        return self.data.shape[0]

    @property
    def cols(self) -> int:
        """Number of timepoints."""
        return self.data.shape[1]

    def at(self, row: int, col: int) -> float:
        return float(self.data[row, col])


# ============================================================================
# Events / stimuli
# ============================================================================

@dataclass
class EventMarker:
    onset:    float  # seconds
    duration: float  # seconds
    value:    float  # stimulus amplitude


@dataclass
class Event:
    name:    str
    markers: list[EventMarker] = field(default_factory=list)


@dataclass
class EventsContainer:
    events: list[Event] = field(default_factory=list)


# ============================================================================
# Biosignals (auxiliary)
# ============================================================================

class AuxiliaryType(Enum):
    UNKNOWN     = 0
    RESPIRATION = 1
    GSR         = 2
    TEMPERATURE = 3
    EX_GA_1     = 4
    EX_GA_2     = 5
    EX_GA_3     = 6
    PPG         = 7
    SP_O2       = 8
    HEARTRATE   = 9


_AUX_NAME_MAP: dict[str, AuxiliaryType] = {
    "Respiration": AuxiliaryType.RESPIRATION,
    "GSR":         AuxiliaryType.GSR,
    "Temperature": AuxiliaryType.TEMPERATURE,
    "ExGa_1":      AuxiliaryType.EX_GA_1,
    "ExGa_2":      AuxiliaryType.EX_GA_2,
    "ExGa_3":      AuxiliaryType.EX_GA_3,
    "PPG":         AuxiliaryType.PPG,
    "SpO2":        AuxiliaryType.SP_O2,
    "Heartrate":   AuxiliaryType.HEARTRATE,
}


@dataclass
class AuxiliaryData:
    label: int            # 0-indexed slot index (AUX1=0, AUX2=1, ...)
    type:  AuxiliaryType
    name:  str
    unit:  str
    data:  np.ndarray = field(default_factory=lambda: np.array([]))
    time:  np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class BiosignalData:
    aux_data: list[AuxiliaryData] = field(default_factory=list)


# ============================================================================
# Top-level SNIRF container  (mirrors snirfcpp/SNIRF.h)
# ============================================================================

@dataclass
class SNIRF:
    """
    Top-level fNIRS data container, mirroring the C++ SNIRF struct in snirf-cpp.

    HDF5 layout:
        /nirs/metaDataTags/     → metadata
        /nirs/probe/            → probe  (sources, detectors, wavelengths, channels)
        /nirs/data1/            → channel_store + time_data
        /nirs/stim1, stim2, …   → events
        /nirs/aux1, aux2, …     → biosignals  (only when metadata.has_wings_generation)
    """
    filepath:      str
    metadata:      Metadata
    probe:         Probe
    channel_store: ChannelDataStore
    time_data:     TimeData
    events:        EventsContainer
    biosignals:    BiosignalData

    # ---- convenience accessors (mirroring C++ SNIRF methods) ----

    def is_file_loaded(self) -> bool:
        return bool(self.filepath)

    def get_sampling_rate(self) -> float:
        return self.time_data.sampling_frequency

    def get_duration(self) -> float:
        return self.time_data.duration

    def get_time(self) -> np.ndarray:
        return self.time_data.time

    def get_wavelengths(self) -> list[int]:
        return self.probe.wavelengths

    def get_source_count(self) -> int:
        return len(self.probe.sources)

    def get_detector_count(self) -> int:
        return len(self.probe.detectors)

    def get_channel_count(self) -> int:
        return self.channel_store.rows

    def has_biosignals(self) -> bool:
        return self.metadata.has_wings_generation

    def get_channel_names(self) -> list[str]:
        """
        Generate channel names in the format 'S{src}-D{det} {wavelength}'.
        First N/2 channels use wavelengths[0], next N/2 use wavelengths[1],
        matching the ParseData1 pairing (hbr_idx first, hbo_idx second).
        """
        n_pairs = len(self.probe.channels)
        wl = self.probe.wavelengths
        names = []
        for ch_id in sorted(self.probe.channels):
            ch = self.probe.channels[ch_id]
            names.append(f"S{ch.source_id}-D{ch.detector_id} {wl[0]}")
        for ch_id in sorted(self.probe.channels):
            ch = self.probe.channels[ch_id]
            names.append(f"S{ch.source_id}-D{ch.detector_id} {wl[1]}")
        return names

    def print(self) -> None:
        print("SNIRF")
        print(f"  filepath           : {self.filepath}")
        print(f"  sampling_frequency : {self.time_data.sampling_frequency:.4f} Hz")
        print(f"  duration           : {self.time_data.duration:.2f} s")
        print(f"  channels (total)   : {self.channel_store.rows}")
        print(f"  timepoints         : {self.channel_store.cols}")
        print(f"  sources            : {self.get_source_count()}")
        print(f"  detectors          : {self.get_detector_count()}")
        print(f"  wavelengths        : {self.probe.wavelengths}")
        print(f"  events             : {sum(len(e.markers) for e in self.events.events)}")
        print(f"  biosignals         : {len(self.biosignals.aux_data)}")
        print(f"  metadata tags      : {len(self.metadata.tags)}")
        print(f"  has_wings_gen      : {self.metadata.has_wings_generation}")

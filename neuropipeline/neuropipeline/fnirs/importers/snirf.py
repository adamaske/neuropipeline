"""
h5py-based SNIRF importer.

Parsing logic mirrors SNIRFLoader.cpp from snirf-cpp:
    ParseMetadata  → _parse_metadata()
    ParseProbe     → _parse_probe()
    ParseTime      → _parse_time()
    ParseData1     → _parse_data1()
    ParseStims     → _parse_stims()
    ParseBiosignals → _parse_biosignals()
"""

import h5py
import numpy as np

from .base import BasefNIRSImporter
from ..snirf_types import (
    SNIRF,
    Metadata, MetadataTag,
    Probe, Optode, OptodeType, Channel, Vec2, Vec3,
    TimeData,
    ChannelDataStore,
    EventsContainer, Event, EventMarker,
    BiosignalData, AuxiliaryData, AuxiliaryType, _AUX_NAME_MAP,
)


# ============================================================================
# Section parsers
# ============================================================================

def _parse_metadata(meta_group: h5py.Group) -> Metadata:
    """Mirrors ParseMetadata() in SNIRFLoader.cpp."""
    meta = Metadata()
    for name in meta_group.keys():
        if name == "wingsGeneration":
            meta.has_wings_generation = True
        try:
            ds = meta_group[name]
            if not isinstance(ds, h5py.Dataset):
                continue
            raw = ds[()]
            if isinstance(raw, bytes):
                value = raw.decode("utf-8")
            elif isinstance(raw, np.ndarray) and raw.dtype.kind in ("S", "O"):
                item = raw.flat[0]
                value = item.decode("utf-8") if isinstance(item, bytes) else str(item)
            else:
                value = "(non-string)"
            meta.tags.append(MetadataTag(name=name, value=value))
        except Exception:
            pass
    return meta


def _read_float_matrix(ds: h5py.Dataset) -> np.ndarray:
    """Read a 2-D float dataset as a (rows, cols) numpy array."""
    dims = ds.shape
    raw = np.empty(dims[0] * dims[1], dtype=np.float64)
    ds.read_direct(raw)
    return raw.reshape(dims[0], dims[1])


def _parse_probe(probe_group: h5py.Group) -> Probe:
    """Mirrors ParseProbe() in SNIRFLoader.cpp."""
    sources:   dict[int, Optode] = {}
    detectors: dict[int, Optode] = {}

    # ---- Detectors ----
    if "detectorPos2D" in probe_group:
        det2D = _read_float_matrix(probe_group["detectorPos2D"])
        det3D = _read_float_matrix(probe_group["detectorPos3D"]) \
            if "detectorPos3D" in probe_group else None

        for i in range(det2D.shape[0]):
            pos2 = Vec2(x=float(det2D[i, 0]), y=float(det2D[i, 1]))
            pos3 = None
            if det3D is not None:
                # y-z swap to match C++ convention
                pos3 = Vec3(x=float(det3D[i, 0]),
                            y=float(det3D[i, 2]),
                            z=float(det3D[i, 1]))
            oid = i + 1  # 1-indexed
            detectors[oid] = Optode(type=OptodeType.DETECTOR, id=oid,
                                    position_2D=pos2, position_3D=pos3)

    # ---- Sources ----
    if "sourcePos2D" in probe_group:
        src2D = _read_float_matrix(probe_group["sourcePos2D"])
        src3D = _read_float_matrix(probe_group["sourcePos3D"]) \
            if "sourcePos3D" in probe_group else None

        for i in range(src2D.shape[0]):
            pos2 = Vec2(x=float(src2D[i, 0]), y=float(src2D[i, 1]))
            pos3 = None
            if src3D is not None:
                pos3 = Vec3(x=float(src3D[i, 0]),
                            y=float(src3D[i, 2]),
                            z=float(src3D[i, 1]))
            oid = i + 1
            sources[oid] = Optode(type=OptodeType.SOURCE, id=oid,
                                  position_2D=pos2, position_3D=pos3)

    # ---- Wavelengths ----
    wavelengths: list[int] = []
    if "wavelengths" in probe_group:
        wl_raw = probe_group["wavelengths"][()]
        wavelengths = sorted(int(w) for w in wl_raw.flat)

    return Probe(channels={}, sources=sources, detectors=detectors,
                 wavelengths=wavelengths)


def _parse_time(time_ds: h5py.Dataset) -> TimeData:
    """Mirrors ParseTime() in SNIRFLoader.cpp."""
    time_vec = time_ds[()].astype(np.float64).ravel()
    total_duration = float(time_vec[-1] - time_vec[0])
    n_intervals = len(time_vec) - 1
    avg_dt = total_duration / n_intervals if n_intervals > 0 else 0.0
    sampling_frequency = 1.0 / avg_dt if avg_dt > 0.0 else 0.0
    return TimeData(time=time_vec,
                    duration=total_duration,
                    sampling_frequency=sampling_frequency)


def _parse_data1(data1_group: h5py.Group, probe: Probe) -> tuple[TimeData, ChannelDataStore]:
    """
    Mirrors ParseData1() in SNIRFLoader.cpp.

    HDF5 layout: dataTimeSeries is [timepoints × channels].
    Transposes to [channels × timepoints] for ChannelDataStore.
    Reads measurementList{i} for i in 1..N/2 to build Channel structs
    (first N/2 = hbr_data, second N/2 = hbo_data).
    """
    time_data = _parse_time(data1_group["time"])

    # Raw matrix: HDF5 stores [timepoints, channels]
    ts_raw = data1_group["dataTimeSeries"][()].astype(np.float64)
    if ts_raw.ndim == 1:
        ts_raw = ts_raw.reshape(-1, 1)
    n_timepoints, n_channels = ts_raw.shape

    if n_channels % 2 != 0:
        raise ValueError(
            f"SNIRF data1 has {n_channels} channels; expected an even number "
            f"(HbR + HbO pairs)."
        )

    # Transpose to [channels, timepoints]
    data_T = ts_raw.T.copy()  # shape (n_channels, n_timepoints)
    store = ChannelDataStore(data=data_T)

    half = n_channels // 2

    # Build Channel structs: first half = hbr, second half = hbo
    for i in range(half):
        hbr_idx = i
        hbo_idx = i + half

        ml_name = f"measurementList{i + 1}"
        if ml_name not in data1_group:
            continue

        ml = data1_group[ml_name]
        source_id   = int(ml["sourceIndex"][()])
        detector_id = int(ml["detectorIndex"][()])

        hbr = data_T[hbr_idx].copy()
        hbo = data_T[hbo_idx].copy()
        hbt = hbo + hbr

        ch = Channel(id=i, source_id=source_id, detector_id=detector_id,
                     hbo_data=hbo, hbr_data=hbr, hbt_data=hbt)
        probe.channels[ch.id] = ch

    return time_data, store


def _parse_stims(nirs_group: h5py.Group) -> EventsContainer:
    """Mirrors ParseStims() in SNIRFLoader.cpp."""
    container = EventsContainer()
    for i in range(1, 1000):
        stim_key = f"stim{i}"
        if stim_key not in nirs_group:
            break

        stim = nirs_group[stim_key]
        if "data" not in stim:
            continue

        raw = stim["data"][()]
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        name = ""
        if "name" in stim:
            raw_name = stim["name"][()]
            if isinstance(raw_name, bytes):
                name = raw_name.decode("utf-8")
            elif isinstance(raw_name, np.ndarray):
                item = raw_name.flat[0]
                name = item.decode("utf-8") if isinstance(item, bytes) else str(item)
            else:
                name = str(raw_name)

        markers: list[EventMarker] = []
        for row in raw:
            if len(row) < 3:
                continue
            markers.append(EventMarker(onset=float(row[0]),
                                       duration=float(row[1]),
                                       value=float(row[2])))

        markers.sort(key=lambda m: m.onset)
        container.events.append(Event(name=name, markers=markers))

    return container


def _parse_biosignals(nirs_group: h5py.Group) -> BiosignalData:
    """Mirrors ParseBiosignals() in SNIRFLoader.cpp."""
    bio = BiosignalData()
    aux_keys = [
        ("aux1", 0), ("aux2", 1), ("aux3", 2),
        ("aux4", 3), ("aux5", 4), ("aux6", 5),
        ("aux7", 6), ("aux8", 7), ("aux9", 8),
    ]
    for group_name, label_idx in aux_keys:
        if group_name not in nirs_group:
            continue
        aux_group = nirs_group[group_name]

        def _read_str(key: str) -> str:
            raw = aux_group[key][()]
            if isinstance(raw, bytes):
                return raw.decode("utf-8")
            if isinstance(raw, np.ndarray):
                item = raw.flat[0]
                return item.decode("utf-8") if isinstance(item, bytes) else str(item)
            return str(raw)

        name      = _read_str("name") if "name" in aux_group else ""
        unit      = _read_str("dataUnit") if "dataUnit" in aux_group else ""
        aux_type  = _AUX_NAME_MAP.get(name, AuxiliaryType.UNKNOWN)
        data_vals = aux_group["dataTimeSeries"][()].astype(np.float64).ravel() \
            if "dataTimeSeries" in aux_group else np.array([])
        time_vals = aux_group["time"][()].astype(np.float64).ravel() \
            if "time" in aux_group else np.array([])

        bio.aux_data.append(AuxiliaryData(
            label=label_idx, type=aux_type,
            name=name, unit=unit,
            data=data_vals, time=time_vals,
        ))
    return bio


# ============================================================================
# Public importer
# ============================================================================

class SNIRFImporter(BasefNIRSImporter):
    """
    Loads a SNIRF (.snirf) file using h5py and returns a SNIRF data object.

    Parsing mirrors the C++ SNIRFLoader.cpp (snirf-cpp project):
        1. metaDataTags  → metadata
        2. probe         → probe (sources, detectors, wavelengths)
        3. data1         → channel_store + time_data + probe.channels
        4. stim{i}       → events
        5. aux{i}        → biosignals  (only if metadata.has_wings_generation)
    """

    def load(self, filepath: str) -> SNIRF:
        print(f"SNIRFImporter: reading {filepath}")

        metadata      = Metadata()
        probe         = Probe(channels={}, sources={}, detectors={}, wavelengths=[])
        channel_store = ChannelDataStore()
        time_data     = TimeData()
        events        = EventsContainer()
        biosignals    = BiosignalData()

        with h5py.File(filepath, "r") as f:
            # Locate the /nirs group (handles /nirs, /nirs1, /nirs2, …)
            nirs_keys = [k for k in f.keys() if k.startswith("nirs")]
            if not nirs_keys:
                raise ValueError(f"No 'nirs' group found in {filepath}")
            nirs_key = "nirs" if "nirs" in nirs_keys else sorted(nirs_keys)[0]
            nirs = f[nirs_key]

            # 1. Metadata
            if "metaDataTags" in nirs:
                metadata = _parse_metadata(nirs["metaDataTags"])

            # 2. Probe
            if "probe" in nirs:
                probe = _parse_probe(nirs["probe"])

            # 3. Data1
            if "data1" in nirs:
                time_data, channel_store = _parse_data1(nirs["data1"], probe)

            # 4. Stims
            events = _parse_stims(nirs)

            # 5. Biosignals (only when wingsGeneration tag is present)
            if metadata.has_wings_generation:
                biosignals = _parse_biosignals(nirs)

        snirf = SNIRF(
            filepath=filepath,
            metadata=metadata,
            probe=probe,
            channel_store=channel_store,
            time_data=time_data,
            events=events,
            biosignals=biosignals,
        )
        print(f"SNIRFImporter: loaded {snirf.channel_store.rows} channels × "
              f"{snirf.channel_store.cols} samples "
              f"@ {snirf.time_data.sampling_frequency:.2f} Hz")
        return snirf

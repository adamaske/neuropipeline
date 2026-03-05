import os

import h5py
import numpy as np

from .base import BasefNIRSExporter
from ..snirf_types import SNIRF, Metadata, Probe, ChannelDataStore, TimeData, EventsContainer, BiosignalData

# Variable-length UTF-8 string dtype used throughout
_STR_DTYPE = h5py.string_dtype(encoding="utf-8")


# ============================================================================
# Section writers  (mirror the section parsers in importers/snirf.py)
# ============================================================================

def _write_str(group: h5py.Group, name: str, value: str) -> None:
    group.create_dataset(name, data=value, dtype=_STR_DTYPE)


def _write_metadata(nirs: h5py.Group, metadata: Metadata) -> None:
    mg = nirs.create_group("metaDataTags")
    for tag in metadata.tags:
        _write_str(mg, tag.name, tag.value)
    if metadata.has_wings_generation:
        _write_str(mg, "wingsGeneration", "1")


def _write_probe(nirs: h5py.Group, probe: Probe) -> None:
    pg = nirs.create_group("probe")

    # Wavelengths
    if probe.wavelengths:
        pg.create_dataset("wavelengths",
                          data=np.array(probe.wavelengths, dtype=np.float64))

    # Sources
    src_ids = sorted(probe.sources)
    if src_ids:
        src2D = np.zeros((len(src_ids), 2), dtype=np.float64)
        src3D_rows = []
        has_3d = any(probe.sources[sid].position_3D is not None for sid in src_ids)

        for i, sid in enumerate(src_ids):
            opt = probe.sources[sid]
            if opt.position_2D is not None:
                src2D[i] = [opt.position_2D.x, opt.position_2D.y]
            if opt.position_3D is not None:
                # Undo the y-z swap applied during import: HDF5 stores [x, z, y]
                p = opt.position_3D
                src3D_rows.append([p.x, p.z, p.y])
            else:
                src3D_rows.append([0.0, 0.0, 0.0])

        pg.create_dataset("sourcePos2D", data=src2D)
        if has_3d:
            pg.create_dataset("sourcePos3D",
                              data=np.array(src3D_rows, dtype=np.float64))

    # Detectors
    det_ids = sorted(probe.detectors)
    if det_ids:
        det2D = np.zeros((len(det_ids), 2), dtype=np.float64)
        det3D_rows = []
        has_3d = any(probe.detectors[did].position_3D is not None for did in det_ids)

        for i, did in enumerate(det_ids):
            opt = probe.detectors[did]
            if opt.position_2D is not None:
                det2D[i] = [opt.position_2D.x, opt.position_2D.y]
            if opt.position_3D is not None:
                p = opt.position_3D
                det3D_rows.append([p.x, p.z, p.y])
            else:
                det3D_rows.append([0.0, 0.0, 0.0])

        pg.create_dataset("detectorPos2D", data=det2D)
        if has_3d:
            pg.create_dataset("detectorPos3D",
                              data=np.array(det3D_rows, dtype=np.float64))


def _write_data1(nirs: h5py.Group, store: ChannelDataStore,
                 time_data: TimeData, probe: Probe) -> None:
    dg = nirs.create_group("data1")

    # dataTimeSeries: HDF5 layout is [timepoints × channels]
    # channel_store.data is [channels × timepoints], so transpose
    ts = store.data.T.astype(np.float64)  # (timepoints, channels)
    dg.create_dataset("dataTimeSeries", data=ts)
    dg.create_dataset("time", data=time_data.time.astype(np.float64))

    # measurementList{i} for i in 1..N/2
    # First N/2 columns = HbR, second N/2 = HbO — one measurementList per pair
    for ch_id in sorted(probe.channels):
        ch = probe.channels[ch_id]
        ml = dg.create_group(f"measurementList{ch_id + 1}")
        ml.create_dataset("sourceIndex",   data=np.int32(ch.source_id))
        ml.create_dataset("detectorIndex", data=np.int32(ch.detector_id))


def _write_stims(nirs: h5py.Group, events: EventsContainer) -> None:
    stim_num = 1
    for event in events.events:
        if not event.markers:
            continue
        sg = nirs.create_group(f"stim{stim_num}")
        _write_str(sg, "name", event.name)
        stim_data = np.array(
            [[m.onset, m.duration, m.value] for m in event.markers],
            dtype=np.float64,
        )
        sg.create_dataset("data", data=stim_data)
        stim_num += 1


def _write_biosignals(nirs: h5py.Group, biosignals: BiosignalData) -> None:
    for i, aux in enumerate(biosignals.aux_data, start=1):
        ag = nirs.create_group(f"aux{i}")
        _write_str(ag, "name", aux.name)
        _write_str(ag, "dataUnit", aux.unit)
        ag.create_dataset("dataTimeSeries", data=aux.data.astype(np.float64))
        ag.create_dataset("time",           data=aux.time.astype(np.float64))


def _write_snirf(f: h5py.File, data: SNIRF) -> None:
    """Write all SNIRF sections into an open HDF5 file handle."""
    nirs = f.create_group("nirs")
    _write_metadata(nirs, data.metadata)
    _write_probe(nirs, data.probe)
    _write_data1(nirs, data.channel_store, data.time_data, data.probe)
    _write_stims(nirs, data.events)
    if data.metadata.has_wings_generation:
        _write_biosignals(nirs, data.biosignals)


# ============================================================================
# Public exporter
# ============================================================================

class SNIRFExporter(BasefNIRSExporter):
    """
    Exports a SNIRF object to a .snirf (HDF5) file, written from scratch.

    Writes:
        /nirs/metaDataTags/     — all metadata tags (+ wingsGeneration flag)
        /nirs/probe/            — source/detector positions (2D + 3D), wavelengths
        /nirs/data1/            — dataTimeSeries, time, measurementList{i}
        /nirs/stim{i}/          — events
        /nirs/aux{i}/           — biosignals (only when has_wings_generation)
    """

    def export(self, data: SNIRF, output_path: str) -> None:
        if os.path.exists(output_path):
            ans = input(f"{output_path} already exists. Overwrite? [Y/N]: ")
            if ans.strip().upper() != "Y":
                print("SNIRFExporter: export cancelled.")
                return

        with h5py.File(output_path, "w") as f:
            _write_snirf(f, data)

        print(f"SNIRFExporter: wrote {output_path}")

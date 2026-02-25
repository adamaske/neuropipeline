from pathlib import Path
import scipy.io

from neuropipeline.eeg import EEGImporter, EEGExporter, EEGData
from neuropipeline.eeg import visualizer

# Paths relative to this file
DATA_DIR = Path(__file__).parent / "data"
STUDY_DIR = DATA_DIR / "eeg_study"
OUTPUT_DIR = DATA_DIR / "eeglab_output"


# ─── Test 1: Single file import ──────────────────────────────────────────────

def test_single_import():
    print("\n=== Test 1: Single file import ===")
    filepath = next(STUDY_DIR.glob("*.hdf5"))
    eeg = EEGImporter.load(str(filepath), source="gRecorder")
    assert isinstance(eeg, EEGData)
    eeg.print()
    print("PASSED\n")
    return eeg


# ─── Test 2: Study folder import ─────────────────────────────────────────────

def test_study_import():
    print("\n=== Test 2: Study folder import ===")
    files = sorted(STUDY_DIR.glob("*.hdf5"))
    assert len(files) > 0, f"No .hdf5 files found in {STUDY_DIR}"
    print(f"Found {len(files)} files:")

    recordings = []
    for filepath in files:
        eeg = EEGImporter.load(str(filepath), source="gRecorder")
        assert isinstance(eeg, EEGData)
        assert eeg.channel_data.shape[0] == eeg.channel_num
        assert eeg.channel_data.shape[1] > 0
        assert len(eeg.channel_names) == eeg.channel_num
        print(f"  {filepath.name}  |  {eeg.channel_num}ch  {eeg.sampling_frequency}Hz  {eeg.get_duration():.1f}s  {len(eeg.feature_onsets)} events")
        recordings.append(eeg)

    print(f"All {len(recordings)} files loaded successfully.")
    print("PASSED\n")
    return recordings


# ─── Test 3: EEGLAB export (single file) ─────────────────────────────────────

def test_eeglab_export(eeg: EEGData, name: str = "test_export"):
    print("\n=== Test 3: EEGLAB export ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    EEGExporter.to_eeglab(eeg, output_folder=str(OUTPUT_DIR), name=name)

    set_path = OUTPUT_DIR / (name + ".set")
    fdt_path = OUTPUT_DIR / (name + ".fdt")
    assert set_path.exists(), f".set file not found: {set_path}"
    assert fdt_path.exists(), f".fdt file not found: {fdt_path}"

    # Inspect the .set struct
    mat = scipy.io.loadmat(str(set_path), simplify_cells=True)
    eeg_struct = mat["EEG"]
    print(f"  srate    : {eeg_struct['srate']}")
    print(f"  nbchan   : {eeg_struct['nbchan']}")
    print(f"  pnts     : {eeg_struct['pnts']}")
    print(f"  xmax     : {eeg_struct['xmax']:.2f}s")
    print(f"  data ref : {eeg_struct['data']}")
    print(f"  .fdt     : {fdt_path.stat().st_size / 1e6:.2f} MB")

    assert float(eeg_struct["srate"]) == eeg.sampling_frequency
    assert int(eeg_struct["nbchan"]) == eeg.channel_num

    print("PASSED\n")


# ─── Test 4: Batch EEGLAB export ─────────────────────────────────────────────

def test_batch_export():
    print("\n=== Test 4: Batch EEGLAB export ===")
    files = sorted(STUDY_DIR.glob("*.hdf5"))
    batch_out = OUTPUT_DIR / "batch"
    batch_out.mkdir(parents=True, exist_ok=True)

    for filepath in files:
        eeg = EEGImporter.load(str(filepath), source="gRecorder")
        EEGExporter.to_eeglab(eeg, output_folder=str(batch_out), name=filepath.stem)
        assert (batch_out / (filepath.stem + ".set")).exists()
        assert (batch_out / (filepath.stem + ".fdt")).exists()
        print(f"  exported: {filepath.stem}")

    print(f"\n{len(files)} files exported to {batch_out}")
    print("PASSED\n")


# ─── Test 5: Visualizer from EEGData object ──────────────────────────────────

def test_visualizer_from_data(eeg: EEGData):
    print("\n=== Test 5: Visualizer from EEGData object ===")
    print("  Opening visualizer (close window to continue)...")
    visualizer.set_marker_dictionary({2: "Trial Start", 4: "Cue", 5: "Move", 6: "Rest"})
    visualizer.set_spectrogram_limits(0.5, 50.0)
    visualizer.set_spectrum_mode("FFT")
    visualizer.open(eeg)
    print("PASSED\n")


# ─── Test 6: Visualizer from filepath string ─────────────────────────────────

def test_visualizer_from_path():
    print("\n=== Test 6: Visualizer from filepath string ===")
    filepath = str(next(STUDY_DIR.glob("*.hdf5")))
    print(f"  Loading and opening: {Path(filepath).name}")
    print("  Opening visualizer (close window to continue)...")
    visualizer.set_spectrogram_method("STFT")
    visualizer.open(filepath)
    print("PASSED\n")


# ─── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    eeg = test_single_import()
    test_study_import()
    test_eeglab_export(eeg, name=next(STUDY_DIR.glob("*.hdf5")).stem)
    test_batch_export()
    print("=== Automated tests passed ===\n")
    test_visualizer_from_data(eeg)
    test_visualizer_from_path()
    print("=== All tests passed ===")

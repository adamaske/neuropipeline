# neuropipeline

`neuropipeline` is a Python package designed for comprehensive analysis of neurophysiological data, specifically Electroencephalography (EEG) and functional Near-Infrared Spectroscopy (fNIRS). It provides tools for preprocessing, analysis, and visualization of both EEG and fNIRS data, streamlining your neuroimaging workflow.

## Features

* **Unified Interface:** A consistent API for both EEG and fNIRS data processing.
* **Preprocessing:**
    * Artifact removal (e.g., ICA, filtering).
    * Data cleaning and interpolation.
    * Channel/probe management.
* **Analysis:**
    * Frequency domain analysis (e.g., power spectral density).
    * Time domain analysis (e.g., event-related potentials/responses).
    * Statistical analysis.
    * fNIRS specific analysis (HbO, HbR, and HbT calculations)
* **Visualization:**
    * Interactive plots for EEG and fNIRS data.
    * Topographic maps.
    * Time-series plots.
    * fNIRS channel location visualization.
* **Extensible Design:** Easily add custom processing and analysis modules.
* **Well-Documented:** Comprehensive documentation with examples and tutorials.

## Installation

```bash
pip install neuropipeline
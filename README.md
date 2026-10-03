# End-to-End EEG Preprocessing and Deep Learning Pipeline

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![MNE-Python](https://img.shields.io/badge/MNE-Python_1.5+-008080.svg)](https://mne.tools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, reproducible Python pipeline for electroencephalography (EEG) signal preprocessing, ocular artifact suppression via Independent Component Analysis (ICA), spectral band-power extraction, and downstream classification with compact convolutional architectures (EEGNet).

---

## Key Features

- **Automated Filtering Pipeline:** Zero-phase bandpass filtering (0.5–45.0 Hz) coupled with notch filtering (50 Hz) for line noise attenuation.
- **ICA-Driven Artifact Removal:** High-pass stabilized FastICA decomposition with automated ocular component detection and suppression.
- **Spectral Feature Extraction:** Frequency band decomposition (Delta, Theta, Alpha, Beta, Gamma) via Welch Power Spectral Density (PSD) estimation.
- **Neural Architecture Integration:** Ready-to-train PyTorch implementation of `EEGNetLight` optimized for low-channel and standard 10–20 montage inputs.
- **Reproducible Demo & Verification:** Unit-tested modules (`pytest`) and a standalone demonstration notebook requiring zero external network downloads.

---

## Repository Structure

```text
eeg-artifact-removal-pipeline/
├── src/
│   ├── preprocess.py      # Bandpass, notch filtering, and FastICA artifact suppression
│   ├── features.py        # Welch PSD band-power extraction (Delta to Gamma)
│   └── model.py           # EEGNetLight PyTorch neural architecture
├── tests/
│   └── test_pipeline.py   # Unit tests verifying shapes, filters, and forward passes
├── demo.ipynb             # End-to-end execution notebook (Synthetic & Real EEG compatible)
├── requirements.txt       # Core dependencies
└── README.md
```

---

## Pipeline Overview

```
Raw EEG Data (.edf / .bdf / Array)
  │
  ├──► [1] Temporal & Notch Filtering (0.5 - 45 Hz + 50 Hz Notch)
  │
  ├──► [2] FastICA Decomposition & EOG Artifact Suppression
  │
  ├──► [3] Welch PSD Spectral Extraction (Delta, Theta, Alpha, Beta, Gamma)
  │
  └──► [4] Tensor Formatting & EEGNet Classification (Logits Output)
```

---

## Installation

Clone the repository and install required packages in a clean virtual environment:

```bash
git clone https://github.com/Avin-Amiri/eeg-artifact-removal-pipeline.git
cd eeg-artifact-removal-pipeline
pip install -r requirements.txt
```

---

## Quickstart

### 1. Interactive Demo (`demo.ipynb`)
Open and run `demo.ipynb` to execute the full pipeline. The notebook generates a synthetic 32-channel EEG session with line noise and ocular blinks, processes the raw signals through all filtering/ICA stages, extracts spectral features, and feeds the formatted tensors into `EEGNetLight`.

### 2. Programmatic Usage on Custom Datasets
The pipeline accepts any standard `mne.io.Raw` object (e.g., loaded from `.edf`, `.bdf`, or `.fif` files):

```python
import mne
import torch
from src.preprocess import EEGPreprocessor
from src.features import extract_band_powers
from src.model import EEGNetLight

# Load your custom EEG recording
raw = mne.io.read_raw_edf("path_to_data.edf", preload=True)

# 1. Preprocessing & ICA
preprocessor = EEGPreprocessor(l_freq=1.0, h_freq=40.0, notch_freq=50.0)
filtered_raw = preprocessor.apply_filters(raw)
ica = preprocessor.fit_ica(filtered_raw, n_components=15)
cleaned_raw = preprocessor.remove_eog_artifacts(filtered_raw, ica=ica)

# 2. Spectral Feature Extraction
features = extract_band_powers(cleaned_raw)

# 3. Model Forward Pass
model = EEGNetLight(num_channels=len(raw.ch_names), num_classes=2, samples=128)
model.eval()

dummy_tensor = torch.randn(1, 1, len(raw.ch_names), 128)
with torch.no_grad():
    logits = model(dummy_tensor)
```

---

## Verification & Unit Testing

Run the test suite to verify component integrity, input/output dimensions, and forward passes across modules:

```bash
python -m pytest tests/ -v
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

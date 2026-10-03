# EEG Artifact Removal & Signal Processing Pipeline

A modular and reproducible Python pipeline designed for preprocessing electroencephalography (EEG) signals, performing independent component analysis (ICA) for artifact attenuation, extracting spectral power bands, and classifying neurophysiological patterns using a lightweight convolutional neural network (EEGNet).

---

## Architecture & Features

- **Preprocessing (`src/preprocess.py`)**: 
  - Bandpass filtering (Butterworth / FIR) and notch filtering (50/60 Hz line-noise suppression).
  - Extended Infomax / FastICA decomposition via MNE-Python for ocular (EOG) and muscular (EMG) artifact rejection.
- **Feature Extraction (`src/features.py`)**:
  - Multichannel spectral power computation across standard clinical bands: Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), and Gamma (30–45 Hz) using Welch's method.
- **Deep Learning Model (`src/model.py`)**:
  - PyTorch implementation of `EEGNetLight` utilizing depthwise separable convolutions to handle spatial-temporal neuroimaging dynamics with minimal parameters.
- **Unit Testing (`tests/test_pipeline.py`)**:
  - Verification of tensor shapes, multi-class handling, and pipeline sanity using `pytest`.

---

## Project Structure

```text
eeg-artifact-removal-pipeline/
├── src/
│   ├── preprocess.py      # Artifact rejection and filtering routines
│   ├── features.py        # Spectral band power extraction (Welch PSD)
│   └── model.py           # EEGNetLight architecture in PyTorch
├── tests/
│   └── test_pipeline.py   # Unit and integration tests
├── demo.ipynb             # Interactive end-to-end demonstration
├── requirements.txt       # Environment dependencies
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed. Recommended environment setup:

```bash
git clone https://github.com/Avin-Amiri/eeg-artifact-removal-pipeline.git
cd eeg-artifact-removal-pipeline
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Tests

Execute the unit test suite via `pytest`:

```bash
python -m pytest tests/ -v
```

### Usage Example

```python
import torch
from src.model import EEGNetLight

# Batch size: 16, Channels: 32, Time samples: 256
batch_size, n_channels, samples = 16, 32, 256
dummy_eeg = torch.randn(batch_size, 1, n_channels, samples)

model = EEGNetLight(num_channels=n_channels, num_classes=2, samples=samples)
predictions = model(dummy_eeg)

print("Output shape:", predictions.shape)  # Expected: torch.Size([16, 2])
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

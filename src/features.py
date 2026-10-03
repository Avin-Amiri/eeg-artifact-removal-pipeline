import mne
import numpy as np


def extract_band_powers(
    raw: mne.io.Raw, sfreq=None
) -> dict[str, np.ndarray]:
    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 45),
    }

    spectrum = raw.compute_psd(method="welch", fmin=0.5, fmax=45.0)
    psds, freqs = spectrum.get_data(return_freqs=True)

    features = {}
    for band_name, (low, high) in bands.items():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        band_power = psds[:, idx_band].mean(axis=1)
        features[band_name] = band_power

    return features

import mne
import numpy as np


class EEGPreprocessor:

    def __init__(self, l_freq=0.5, h_freq=45.0, notch_freq=50.0):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.ica = None

    def apply_filters(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw_filtered = raw.copy()
        if self.notch_freq:
            raw_filtered.notch_filter(
                freqs=self.notch_freq, verbose=False, picks="eeg"
            )
        raw_filtered.filter(
            l_freq=self.l_freq, h_freq=self.h_freq, verbose=False, picks="eeg"
        )
        return raw_filtered

    def fit_ica(self, raw: mne.io.Raw, n_components=15, random_state=42):
        self.ica = mne.preprocessing.ICA(
            n_components=n_components,
            random_state=random_state,
            method="fastica",
            max_iter="auto",
        )
        raw_ica_ready = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        self.ica.fit(raw_ica_ready, picks="eeg")
        return self.ica

    def remove_eog_artifacts(
        self, raw: mne.io.Raw, eog_channel=None
    ) -> mne.io.Raw:
        if self.ica is None:
            raise ValueError("First run fit_ica before cleaning.")

        raw_clean = raw.copy()

        if eog_channel and eog_channel in raw.ch_names:
            eog_indices, _ = self.ica.find_bads_eog(raw, ch_name=eog_channel)
            self.ica.exclude = eog_indices
        else:
            frontal_chs = [
                ch
                for ch in ["Fp1", "FP1", "Fp2", "FP2", "AF3", "AF4"]
                if ch in raw.ch_names
            ]
            if frontal_chs:
                eog_indices, _ = self.ica.find_bads_eog(
                    raw, ch_name=frontal_chs[0]
                )
                self.ica.exclude = eog_indices
        
        self.ica.apply(raw_clean)
        return raw_clean

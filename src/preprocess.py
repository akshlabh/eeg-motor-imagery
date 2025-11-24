# src/preprocess.py
import mne
import os
from mne.preprocessing import ICA

DEFAULT_RUNS = [3,4,5,6,7,8,9,10,11,12,13,14]  # PhysioNet contains runs 1..14; adapt if needed

def load_eeg_data(subject_folder, runs=None, file_ext=".edf"):
    runs = runs or DEFAULT_RUNS
    print(f"📌 Loading data for: {os.path.basename(subject_folder)}")
    files = [
        os.path.join(subject_folder, f)
        for f in os.listdir(subject_folder)
        if f.endswith(file_ext) and any(f"R{r:02d}" in f for r in runs)
    ]
    if not files:
        raise FileNotFoundError("Motor imagery runs not found in folder: " + subject_folder)
    raws = []
    for f in sorted(files):
        print("  - reading", f)
        raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
        raws.append(raw)
    raw = mne.io.concatenate_raws(raws)
    raw.filter(8.0, 30.0, fir_design="firwin", verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    return raw


def clean_data(raw, n_components=15, random_state=97, apply_ica=False, eog_ch=None):
    """
    Fit ICA and optionally apply it, using FP1/FP2 as EOG proxy if no channel is provided.
    """
    print("🧹 Fitting ICA...")
    ica = ICA(n_components=n_components, max_iter="auto", random_state=random_state, verbose=False)
    # CRITICAL: High-pass filter before ICA fit to remove low-frequency drifts
    raw_for_ica = raw.copy().filter(1.0, None, fir_design="firwin", verbose=False) 
    ica.fit(raw_for_ica, verbose=False)

    raw_clean = raw.copy()
    if apply_ica:
        eog_ch_to_use = eog_ch
        
        # --- NEW LOGIC: Use a frontal channel proxy for EOG if not specified ---
        if eog_ch_to_use is None:
            # Check for standard frontal channels (often AFz, Fp1, Fp2 in 10-20 system)
            # We'll use Fp1 as a common reliable proxy channel for blinks.
            # If the channel list is different, adjust this name.
            if 'Fp1' in raw.ch_names:
                eog_ch_to_use = 'Fp1'
            elif raw.ch_names and raw.ch_names[0].startswith('EEG'):
                # Fallback: Check if the raw data has channel names
                pass # If no EOG channel is found, MNE finds components based on correlation to frontal channels
        
        if eog_ch_to_use:
            try:
                # Find EOG components using the specified/proxied channel
                eog_inds, scores = ica.find_bads_eog(raw_for_ica, ch_name=eog_ch_to_use)
                ica.exclude = list(eog_inds)
                print(f"  → Excluding ICA components by EOG ({eog_ch_to_use}): {ica.exclude}")
                ica.apply(raw_clean)
            except Exception as e:
                print(f"  → Auto EOG detection failed on channel {eog_ch_to_use}:", e)
        else:
            # If still no EOG channel/proxy, we cannot automate artifact removal
            print("  → ICA fitted but not applied: Could not identify EOG channel/proxy.")

    return raw_clean, ica


def create_epochs(cleaned_raw, tmin=0.0, tmax=2.5, picks=None, baseline=None):
    """
    Create epochs from cleaned raw using annotations. Attempts to map 'left'/'right' labels.
    Returns an mne.Epochs object.
    """
    print("⏱ Creating epochs from annotations...")
    events, event_id = mne.events_from_annotations(cleaned_raw, verbose=False)

    mapping = {}
    for name, eid in event_id.items():
        low = name.lower()
        # map common physionet labels (T0/T1/T2) or text
        if "t0" in low or "rest" in low or "blank" in low:
            mapping["rest"] = eid
        elif "t1" in low or "left" in low or "l" == low:
            mapping["left"] = eid
        elif "t2" in low or "right" in low or "r" == low:
            mapping["right"] = eid
        elif "both" in low or "feet" in low:
            mapping.setdefault("both", eid)

    if not mapping:
        keys = list(event_id.keys())
        if len(keys) >= 2:
            mapping = {keys[0]: event_id[keys[0]], keys[1]: event_id[keys[1]]}
        else:
            mapping = event_id

    epochs = mne.Epochs(cleaned_raw, events, event_id=mapping, tmin=tmin, tmax=tmax,
                        picks=picks, preload=True, baseline=baseline, verbose=False)
    print(f"  → Created {len(epochs)} epochs with mapping: {mapping}")
    return epochs

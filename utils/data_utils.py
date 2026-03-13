from pathlib import Path
import numpy as np
import mne


def _read_subject(path):
    """
    Load and preprocess EEG data for a single subject from a GDF file.

    Preprocessing pipeline (applied independently per subject):
        1. Load raw GDF file
        2. Retain all 22 EEG channels and drop 3 EOG channels
        3. Re-reference to the average reference
        4. Apply a low-pass FIR filter (0–38 Hz)
        5. Segment into 4 second epochs
        6. Remap event labels to zero-indexed integers (0–3)

    Args:
        path (str or Path): Path to the .gdf file for one subject

    Returns:
        X (np.ndarray): Epochs array of shape (n_epochs, 22, 1001)
        y (np.ndarray): Zero-indexed integer class labels of shape (n_epochs,)
    """
    raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    raw.drop_channels([ch for ch in raw.ch_names if "EOG" in ch])
    raw.set_eeg_reference("average")
    raw.filter(0, 38, fir_design="firwin")

    # Parse annotations embedded in the GDF file into discrete events
    events, event_dict = mne.events_from_annotations(raw)
    event_id = {}

    # Match annotation labels containing the class codes
    for key, val in event_dict.items():
        if "769" in key or "770" in key or "771" in key or "772" in key:
            event_id[key] = val

    # Segment recordings into 4 second epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0,
        tmax=4,
        baseline=None,
        preload=True,
        on_missing="warn",
        verbose=False,
    )

    # Extract epoch data as a 3D array: (n_epochs, 22, 1001)
    X = epochs.get_data(copy=False)
    # Extract the event code (class label) for each epoch from the events array
    y = epochs.events[:, -1]

    # Remap event codes to consecutive zero-indexed integers (0–3)
    unique_labels = np.unique(y)
    label_map = {label: i for i, label in enumerate(sorted(unique_labels))}
    y = np.array([label_map[label] for label in y])

    return X, y


def load_data(data_dir="data"):
    """
    Load and concatenate EEG data from all 9 subjects in the BCI Competition IV-2a
    training set.

    Preprocessing is applied independently to each subject's recording via
    _read_subject before pooling into a single dataset.

    Expects files named A01T.gdf-A09T.gdf inside `data_dir`.

    Args:
        data_dir (str or Path): Directory containing the .gdf subject files.
                                Defaults to "data".

    Returns:
        X_all (np.ndarray): Concatenated epochs of shape (total_epochs, 22, 1001),
                            cast to float32.
        y_all (np.ndarray): Concatenated class labels of shape (total_epochs,),
                            with values in {0, 1, 2, 3}.
    """
    X_list, y_list = [], []

    for i in range(1, 10):
        file = Path(data_dir) / f"A0{i}T.gdf"
        X, y = _read_subject(file)
        X_list.append(X)
        y_list.append(y)

    # Cast to float32 for PyTorch compatibility
    X_all = np.concatenate(X_list, axis=0).astype("float32")
    y_all = np.concatenate(y_list, axis=0)

    return X_all, y_all

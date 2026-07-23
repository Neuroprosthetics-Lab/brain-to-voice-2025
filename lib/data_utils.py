# ==================================================================================
# Data utilities for brain-to-voice
# Maitreyee Wairagkar, 2025
# ==================================================================================

from functools import lru_cache

import json
import random
from typing import Any, Optional

import numpy as np
import scipy.ndimage

from lib import training_utils, speech_processor


def load_lpc_norm(lpc_norm_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load LPCNet feature normalisation parameters from a JSON file

    Args:
        lpc_norm_path : path to the JSON file containing 'lpcnet_min' and 'lpcnet_max' arrays

    Returns:
        min_lpc_feat : per-feature minimum values as a float32 array
        max_lpc_feat : per-feature maximum values as a float32 array
    """

    with open(lpc_norm_path) as f:
        lpc_dict = json.load(f)
    min_lpc_feat = np.array(lpc_dict["lpcnet_min"], dtype=np.float32)
    max_lpc_feat = np.array(lpc_dict["lpcnet_max"], dtype=np.float32)

    return min_lpc_feat, max_lpc_feat

    
def get_sliding_windows(
    dat: np.ndarray, 
    window_len: int, 
    step_size: int
) -> np.ndarray:
    """
    Extract sliding windows from a neural data array

    Args:
        dat        : input array of shape (time,) or (time, channels)
        window_len : number of time bins per window
        step_size  : number of bins to advance between consecutive windows

    Returns:
        windowed_trial : sliding window view of shape (n_windows, window_len, channels)
    """

    # Define shape of the sliced trial window
    dim = np.ndim(dat)
    y_len = dat.shape[1] if dim == 2 else 1
    window_shape = (window_len, y_len)

    # Implement sliding windows
    windowed_trial = np.lib.stride_tricks.sliding_window_view(
        dat, window_shape=window_shape, writeable=True
    )[::step_size, :]

    return windowed_trial


def scale_between_minus_one_to_one(
    dat: np.ndarray,
    min_val: np.ndarray,
    max_val: np.ndarray,
) -> np.ndarray:
    """
    Scale LPCNet features to the range [-1, 1] using provided min and max values.

    Args:
        dat     : input feature array to scale
        min_val : per-feature minimum values used for normalisation
        max_val : per-feature maximum values used for normalisation

    Returns:
        x_norm : scaled feature array in the range [-1, 1]
    """
 
    x_norm = 2 * ((dat - min_val) / (max_val - min_val)) - 1

    return x_norm


def scale_to_original_range(
    dat: np.ndarray,
    min_val: np.ndarray,
    max_val: np.ndarray,
) -> np.ndarray:
    """
    Rescale LPCNet features from [-1, 1] back to their original range using the provided min and max values

    Args:
        dat     : scaled feature array in the range [-1, 1]
        min_val : per-feature minimum values used during original normalisation
        max_val : per-feature maximum values used during original normalisation

    Returns:
        x_up : feature array rescaled to the original value range
    """

    x_up = (((dat + 1) / 2) * (max_val - min_val)) + min_val

    return x_up


def get_speech_features_from_single_trial(
    lpc_feat_trial: np.ndarray,
    min_lpc_feat: np.ndarray,
    max_lpc_feat: np.ndarray,
    n_cepstral_coeff: int = 18,
    smoothing: bool = True,
    smoothing_sigma: float = 2.0,
) -> np.ndarray:
    """
    Extract and process LPCNet features from a single trial by reducing
    dimensionality, scaling to [-1, 1], and optionally smoothing

    Args:
        lpc_feat_trial   : raw LPCNet feature array of shape (time, 20)
        min_lpc_feat     : per-feature minimum values for normalisation
        max_lpc_feat     : per-feature maximum values for normalisation
        n_cepstral_coeff : number of cepstral coefficients to retain (default is 18, no reduction)
        smoothing        : whether to apply Gaussian smoothing along the time axis
        smoothing_sigma  : standard deviation for Gaussian kernel

    Returns:
        dim_red_feats : processed feature array of shape (time, n_cepstral_coeff + 2)
    """
    # Reduce LPCNet feature dimensions
    dim_red_feats = speech_processor.lpcnet_features_dimensionality_reduction(
        lpc_feat_trial, n_cepstral_coeff=n_cepstral_coeff
    )

    # Scale LPCNet features between -1 and 1
    dim_red_feats = scale_between_minus_one_to_one(dim_red_feats, min_lpc_feat, max_lpc_feat)

    # Smooth the LPCNet features assuming that smooth functions are better to predict:
    if smoothing:
        dim_red_feats = scipy.ndimage.gaussian_filter1d(dim_red_feats, smoothing_sigma, axis=0)

    return dim_red_feats


def format_single_trial(
    neural_trial: dict,
    window_len: int,
    step_size: int,
    min_lpc_feat: Optional[np.ndarray] = None,
    max_lpc_feat: Optional[np.ndarray] = None,
    pre_formatted_Y: Optional[np.ndarray] = None,
    augment: bool = False,
    **aug_kwargs: Any,
) -> dict:
    """
    Format a single neural trial into sliding windows paired with processed
    speech features required for model training. Optionally augments neural
    data with noise before formatting.

    Args:
        neural_trial    : trial dict with keys 'X' (neural data), 'Y' (LPCNet features),
                        'session', and 'session_num'
        window_len      : number of time bins per sliding window
        step_size       : number of bins to advance between consecutive windows
        min_lpc_feat    : per-feature minimum values for LPCNet normalisation.
                        Required if pre_formatted_Y is not provided
        max_lpc_feat    : per-feature maximum values for LPCNet normalisation.
                        Required if pre_formatted_Y is not provided
        pre_formatted_Y : pre-processed target speech features to reuse from the
                        original trial. If provided, skips speech feature processing
        augment         : whether to augment neural data with noise before formatting
        **aug_kwargs    : optional augmentation parameters (sd_gaussian_noise,
                        sd_constant_offset, offset_aug_factor_for_thresh,
                        sd_cumulative_noise)

    Returns:
        dat_dict : formatted trial dict with keys 'X', 'Y', 'session', 'session_num'
    """
    # Optionally augment neural data
    neural_data = training_utils.augment_neural_data(neural_trial["X"], **aug_kwargs) if augment else neural_trial["X"]

    # Divide neural data into overlapping sliding windows
    X_feature_mat = np.squeeze(
        get_sliding_windows(neural_data, window_len=window_len, step_size=step_size)
    )

    # Use pre-formatted Y if provided, otherwise process from scratch
    if pre_formatted_Y is not None:
        Y_feature_mat = pre_formatted_Y
    else:
        Y_feature_mat = get_speech_features_from_single_trial(
            neural_trial["Y"], min_lpc_feat, max_lpc_feat
        )
        # Make Y same length as X starting from the end of first window
        Y_feature_mat = Y_feature_mat[-X_feature_mat.shape[0]:, :]

    return {
        "X": X_feature_mat,
        "Y": Y_feature_mat,
        "session": neural_trial["session"],
        "session_num": neural_trial["session_num"],
    }


def shuffle_trials_sessionwise(
    trial_dict: dict,
    verbose: bool = True,
) -> dict:
    """
    Shuffle trials within each session independently, preserving session boundaries

    Args:
        trial_dict : dict of trials keyed by integer index, each containing a 'session' key
        verbose    : whether to log unique sessions and index ranges

    Returns:
        shuf_dict : trial dict with the same structure but trials shuffled within each session
    """

    trial_list = list(trial_dict.items())

    # Get list of independent sessions
    sessions_list = [trial[1]["session"] for trial in trial_list]

    unique_sessions = list(set(sessions_list))
    if verbose:
        print(f"Unique sessions: {unique_sessions}")

    for sess in unique_sessions:
        sess_idx = [i for i in range(len(sessions_list)) if sessions_list[i] == sess]
        if verbose:
            print(
                f"{sess}: idx_start {sess_idx[0]} idx_end {sess_idx[-1]}, {len(trial_list[sess_idx[0]:sess_idx[-1]+1])}, len_sess_idx: {len(sess_idx)}"
            )

        # Randomly shuffle these indices
        trial_list[sess_idx[0] : sess_idx[-1] + 1] = random.sample(
            trial_list[sess_idx[0] : sess_idx[-1] + 1],
            len(trial_list[sess_idx[0] : sess_idx[-1] + 1]),
        )

    shuffle_dict = {n: dict(trial[1]) for n, trial in enumerate(trial_list)}

    return shuffle_dict

@lru_cache(maxsize=None)
def _load_electrode_mapping(electrode_map_file: str) -> np.ndarray:
    """
    Load and cache the electrode mapping from a JSON file. Cached so repeated
    calls to reorder_channels() with the same path don't re-read the file.

    Args:
        electrode_map_file : path to the JSON file containing the electrode_mapping array

    Returns:
        electrode_mapping : array of electrode indices for reordering channels
    """
    with open(electrode_map_file) as f:
        electrode_mapping_dict = json.load(f)

    return np.array(electrode_mapping_dict["electrode_mapping"])


def reorder_channels(
        neural_trial: np.ndarray, 
        electrode_mapping_file: str = "dependencies/T15_electrode_mapping_256.json",
) -> np.ndarray:
    """
    Reorder channels according to the older ordering for inference using models provided with the data for t15_day025 to t15_day195.
    Note: No reordering is required for new models trained using the provided data or using the provided models for sessions after t15_day195.
    This is because, the older sessions (t15_day025 to t15_day195) had a different electrode mapping than the newer sessions when the provided models were trained iteratively. 
    This ordering is deprecated and is only used for inference on older sessions using the provided models.

    Args:
        neural_trial           : Neural data segment to be reordered.
        electrode_mapping_file : Path to the JSON file containing the electrode mapping.

    Returns:
        np.ndarray: Reordered neural data segment according to the older electrode mapping.
    """

    electrode_mapping = _load_electrode_mapping(electrode_mapping_file)
    channel_mapping = electrode_mapping - 1  # 0-indexed

    # undo final mapping
    step_back_one = neural_trial[:, channel_mapping]

    # map to original (older deprecated) electrode ordering
    original = step_back_one[:, channel_mapping]

    return original
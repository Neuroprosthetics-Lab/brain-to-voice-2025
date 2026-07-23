# ==================================================================================
# Training utilities for brain-to-voice
# Maitreyee Wairagkar, 2025
# ==================================================================================

from typing import Generator

import numpy as np

_RNG = np.random.default_rng()


def augment_neural_data(
    dat: np.ndarray, 
    sd_gaussian_noise: float=1.2,
    sd_constant_offset: float=0.6,
    offset_aug_factor_for_thresh: float=0.67,
    sd_cumulative_noise: float=0.02,
) -> np.ndarray:
    """
    Augment neural data by adding Gaussian noise, constant offset, and cumulative noise.
    If data has both spikepow and threshold crossings, they must be concatenated in dat.

    Args:
        dat                          : single trial neural data of shape (bins, channels) where channels
                                       are spikepow channels concatenated with threshold crossing channels (float32)
        sd_gaussian_noise            : standard deviation of per-sample Gaussian noise
        sd_constant_offset           : standard deviation of the constant offset noise
        offset_aug_factor_for_thresh : scaling factor applied to the constant offset for
                                       threshold crossing channels
        sd_cumulative_noise          : standard deviation of the per-step cumulative noise


    Returns:
        aug_dat : single trial augmented with noise of same shape as input (float32)
    """

    T, C = dat.shape

    # Gaussian noise
    gaussian_noise = _RNG.standard_normal(size=(T, C), dtype=np.float32) * sd_gaussian_noise

    # Constant offset - draw once for spike pow, scale for thresh
    constant_offset_h = _RNG.standard_normal(size=(1, C // 2), dtype=np.float32) * sd_constant_offset
    constant_offset = np.concatenate(
        [constant_offset_h, constant_offset_h * offset_aug_factor_for_thresh], axis=1
    )

    # Cumulative noise
    cum_noise = np.cumsum(_RNG.standard_normal(size=T, dtype=np.float32) * sd_cumulative_noise)[
        :, np.newaxis
    ]

    aug_dat = dat + gaussian_noise + constant_offset + cum_noise

    return aug_dat


def make_single_batch(
    dat: dict,
    trials_in_this_batch: tuple[int, int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract trials to use in this batch from the given trial indices

    Args:
        dat                  : dict of all formatted trials keyed by integer index
        trials_in_this_batch : tuple of (trial_indices, session_num) for this batch
        batch_size           : number of samples to include in the batch

    Returns:
        batch_x     : neural input features of shape (batch_size, time_steps, channels)
        batch_y     : target LPCNet features of shape (batch_size, n_features)
        session_num : session index array of shape (batch_size, 1)
    """

    trials = [dat[tr] for tr in trials_in_this_batch[0]]

    # Concatenate trials
    X_dat, Y_dat = concatenate_feature_windows(trials)

    # Extract single batch from concatenated trials
    batch_x = X_dat[:batch_size]
    batch_y = Y_dat[:batch_size]

    curr_session = [trials_in_this_batch[1]] * batch_size
    session_num = np.array(curr_session, dtype="int32")
    session_num = session_num[:, np.newaxis]

    return batch_x, batch_y, session_num


def make_batches_sessionwise(
    dat: dict,
    batch_size: int,
    verbose: bool = True,
) -> list[list]:
    """
    Generate trial index groupings for batches where all trials in a batch
    belong to the same session

    Args:
        dat        : dict of all formatted trials keyed by integer index
        batch_size : number of samples per batch
        verbose    : whether to log session and batch information

    Returns:
        trials_to_make_batches : list of [tuple(trial_indices), session_num] entries,
                                 one per batch
    """

    trials_to_make_batches = []

    # Get list of independent sessions
    sessions_list = []
    session_num_list = []
    all_trial_sessions = []
    for i in range(len(dat)):
        all_trial_sessions.append(dat[i]["session"])

        if dat[i]["session"] not in sessions_list:
            sessions_list.append(dat[i]["session"])
            session_num_list.append(dat[i]["session_num"])

    if verbose:
        print(f"Sessions: {sessions_list} session nums: {session_num_list} num trials: {len(all_trial_sessions)}")

    for sess_num, sess in enumerate(sessions_list):

        sess_idx = [
            i for i in range(len(all_trial_sessions)) if all_trial_sessions[i] == sess
        ]

        if verbose:
            print(f"{sess}: idx_start {sess_idx[0]} idx_end {sess_idx[-1]}")

        used_trial_idx = sess_idx[0]

        # Generate batches from these trials
        while used_trial_idx < sess_idx[-1]:

            # Compute number of trials required for generating this batch
            trials_in_batch = []
            curr_size = 0
            curr_tr = used_trial_idx
            cnt = 0
            while curr_size < batch_size:

                curr_size += dat[curr_tr]["X"].shape[0]

                trials_in_batch.append(curr_tr)
                used_trial_idx = curr_tr
                curr_tr += 1
                cnt += 1

            if cnt == 1:
                used_trial_idx += 1

            trials_to_make_batches.append([tuple(trials_in_batch), sess_num])

    return trials_to_make_batches


def batch_generator_on_the_fly(
    trials_in_batch: list[list],
    dat: dict,
    batch_size: int,
) -> Generator[tuple[list[np.ndarray], np.ndarray], None, None]:
    """
    Generate a new batch for training

    Args:
        trials_in_batch : list of [tuple(trial_indices), session_num] entries from make_batches_sessionwise
        dat             : dict of all formatted trials keyed by integer index
        batch_size      : number of samples per batch

    Returns:
        X       : list of [batch_x, session_num] arrays as model inputs
        batch_y : target LPCNet features of shape (batch_size, n_features)
    """

    cnt = 0

    while True:

        batch_x, batch_y, this_sess_num = make_single_batch(
            dat, trials_in_batch[cnt], batch_size
        )

        X = [batch_x, this_sess_num]
        yield X, batch_y

        cnt += 1

        if cnt == len(trials_in_batch):
            cnt = 0


def concatenate_feature_windows(
    dat: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate feature windows from multiple trials into single arrays

    Args:
        dat : list of trial dicts each containing 'X' (neural features) and
            'Y' (LPCNet target features) arrays

    Returns:
        x_dat : concatenated neural input features of shape (total_samples, time_steps, channels)
        y_dat : concatenated target LPCNet features of shape (total_samples, n_features)
    """

    n_features_x = dat[0]["X"].shape[1:]
    n_features_y = dat[0]["Y"].shape[1]

    n_samples = sum([trial["X"].shape[0] for trial in dat])

    # concatenate shape tuples by unpacking the shape tuple
    x_dat = np.zeros((n_samples, *n_features_x), dtype="float32")
    y_dat = np.zeros((n_samples, n_features_y), dtype="float32")

    curr_len = 0
    for tr in range(len(dat)):
        tr_len = dat[tr]["X"].shape[0]

        x_dat[curr_len : curr_len + tr_len] = dat[tr]["X"]
        y_dat[curr_len : curr_len + tr_len] = dat[tr]["Y"]

        curr_len += tr_len

    return x_dat, y_dat

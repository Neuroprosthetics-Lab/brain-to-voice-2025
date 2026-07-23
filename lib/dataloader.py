# ==================================================================================
# Functions to load and prepare neural data and corresponding LPCNet targets for brain-to-voice decoder training.
# Maitreyee Wairagkar, 2025
# ==================================================================================

import os
import random
from typing import Optional

import numpy as np
import scipy.io
import yaml

from lib import data_utils

def get_sessions(sessions_path: str) -> tuple[list[str], list[int]]:
    """
    Load session names and their day embedding layer indices from a sessions YAML file

    Args:
        sessions_path : path to the sessions YAML file containing session and session_layer entries

    Returns:
        sessions       : list of session folder name strings in file order
        session_layers : list of integer layer indices aligned to sessions
    """
    with open(sessions_path) as f:
        data = yaml.safe_load(f)
    sessions = [s["session"] for s in data["sessions"]]
    session_layers = [s["session_layer"] for s in data["sessions"]]

    return sessions, session_layers


def load_neural_trials(
    neural_data_path: str,
    training_sessions: list[str],
    session_layers: list[int],
) -> list[dict]:
    """
    Load all neural and speech feature trials from disk for the given sessions

    Args:
        neural_data_path  : path to the directory containing session folders
        training_sessions : list of session folder name strings to load
        session_layers    : integer layer index for each session, aligned by position

    Returns:
        neural_trials : list of trial dicts with keys 'X' (concatenated spikepow and
                        threshcross of shape T x 512), 'Y' (LPCNet target features of
                        shape T x 20), 'trial', 'session', 'session_num', (integer 
                        layer index for this session), and 'sentence' (cue text)
    """
    neural_trials = []

    for sess_num, sess in enumerate(training_sessions):

        neur_mat = scipy.io.loadmat(
            os.path.join(neural_data_path, sess, f"{sess}_neural_data.mat"),
            squeeze_me=True,
        )
        lpc_mat = scipy.io.loadmat(
            os.path.join(neural_data_path, sess, f"{sess}_target_speech_feats.mat"),
            squeeze_me=True,
        )

        spikepow_block = neur_mat["spikepow_trials"]
        threshcross_block = neur_mat["threshcross_trials"]
        cues_block = neur_mat["cues"]
        lpcnet_block = lpc_mat["target_speech_feats_word"]

        for tr in range(len(spikepow_block)):
            trial_dict = {
                "X": np.concatenate([spikepow_block[tr], threshcross_block[tr]], axis=1),
                "Y": lpcnet_block[tr],
                "trial": tr,
                "session": sess,
                "session_num": session_layers[sess_num],
                "sentence": cues_block[tr].strip(),
            }
            neural_trials.append(trial_dict)

        print(f"Loaded {len(spikepow_block)} trials from session {sess}")

    print(f"All data loaded: {len(neural_trials)} trials\n")
    return neural_trials


def make_train_val_split(
    total_trials: int,
    validation_range_start: int,
    validation_range_end: int,
    n_validation_trials: int,
    seed: Optional[int] = None,
) -> tuple[list[int], list[int]]:
    """
    Randomly sample validation trial indices from a fixed range and assign
    all remaining indices to the training set

    Args:
        total_trials           : total number of trials across all sessions
        validation_range_start : start index of the range to sample validation trials from
        validation_range_end   : end index of the range to sample validation trials from
        n_validation_trials    : number of validation trials to randomly sample
        seed                   : optional seed for reproducible validation sampling.
                                 If None, sampling is different on every call.

    Returns:
        training_idx   : list of trial indices assigned to the training set
        validation_idx : list of trial indices assigned to the validation set
    """

    rng = random.Random(seed)

    validation_idx = rng.sample(
        range(validation_range_start, validation_range_end),
        k=n_validation_trials,
    )
    validation_idx = sorted(validation_idx)  # keep session blocks contiguous downstream

    training_idx = [x for x in range(total_trials) if x not in validation_idx]

    print(
        f"Training trials: {len(training_idx)} | Validation trials: {len(validation_idx)}\n"
    )

    return training_idx, validation_idx


def assemble_trials(
    neural_trials: list[dict],
    trial_idx: list[int],
    time_steps: int,
    step_size: int,
    min_lpc_feat: np.ndarray,
    max_lpc_feat: np.ndarray,
    n_augmentations: int = 0,
    aug_cfg: Optional[dict] = None,
    shuffle: bool = False,
) -> dict:
    """
    Format trials into sliding windows paired with processed speech features.
    Optionally augments training trials and shuffles sessionwise.

    Args:
        neural_trials   : list of raw trial dicts as returned by load_neural_trials
        trial_idx       : list of trial indices to include
        time_steps      : number of time bins per sliding window
        step_size       : number of bins to advance between consecutive windows
        min_lpc_feat    : per-feature minimum values for LPCNet normalisation
        max_lpc_feat    : per-feature maximum values for LPCNet normalisation
        n_augmentations : number of noise-augmented copies to generate per trial.
                          Default is 0 (no augmentation)
        aug_cfg         : optional dict of augmentation noise parameters
        shuffle         : whether to shuffle trials sessionwise after assembly

    Returns:
        dat : dict of formatted trials keyed by sequential integer indices
    """
    dat = []

    for tr_count, tr in enumerate(trial_idx):
        dat.append(data_utils.format_single_trial(
            neural_trials[tr], time_steps, step_size, min_lpc_feat, max_lpc_feat
        ))
        print(
            f"Processing trial {tr_count + 1}/{len(trial_idx)} "
            f"neural shape: {dat[-1]['X'].shape} "
            f"target shape: {dat[-1]['Y'].shape}",
            end="\r", flush=True,
        )
        
        # Optionally add augmented version of the trial
        for _ in range(n_augmentations):
            dat.append(data_utils.format_single_trial(
                neural_trials[tr],
                time_steps,
                step_size,
                pre_formatted_Y = dat[-1]["Y"],
                augment = True,
                **(aug_cfg or {}),  # unpacks aug_cfg if provided, empty dict uses function defaults
            ))

    print()

    dat = dict(zip(range(len(dat)), dat))

    # Optionally shuffle sessionwise after assembly
    if shuffle:
        dat = data_utils.shuffle_trials_sessionwise(dat, verbose=False)
        print(f"Trials shuffled sessionwise: {len(dat)}")

    return dat


def load_all_data(cfg: dict) -> tuple[dict, dict, list[str], list[int]]:
    """
    Top-level dataloader function. Loads LPC normalisation
    parameters, discovers sessions, loads raw trials, splits into train and
    validation sets, and assembles formatted datasets (including augmented training trials).

    Args:
        cfg : dict parsed config (from config.yaml).

    Returns:
        training_dat   : dict of formatted + augmented training trials
        validation_dat : dict of formatted validation trials
        sessions       : list of session name strings
        session_layers : list of integer layer indices
    """

    paths = cfg["paths"]
    data = cfg["data"]
    aug_cfg = cfg["data"].get("augmentation_params", None) # optional augmentation parameters

    # LPC normalisation parameters
    min_lpc_feat, max_lpc_feat = data_utils.load_lpc_norm(paths["lpc_norm_path"])

    # 1. Get all training sessions
    sessions, session_layers = get_sessions(paths["training_data"])
    print(f"Number of sessions: {len(sessions)}\n{sessions}")

    # 2. Load raw neural trials
    neural_trials = load_neural_trials(
        paths["neural_data_path"], sessions, session_layers
    )

    # 3. Get train / val split
    training_idx, validation_idx = make_train_val_split(
        total_trials=data["total_trials"],
        validation_range_start=data["validation_range_start"],
        validation_range_end=data["validation_range_end"],
        n_validation_trials=data["n_validation_trials"],
        seed=data.get("random_seed", None)
    )

    # 4. Assemble formatted training and validation datasets
    print(f"Preparing training trials ...")
    training_dat = assemble_trials(
        neural_trials,
        training_idx,
        time_steps=data["time_steps"],
        step_size=data["step_size"],
        min_lpc_feat=min_lpc_feat,
        max_lpc_feat=max_lpc_feat,
        n_augmentations=data["n_augmentations"],
        aug_cfg= aug_cfg,
        shuffle= True,
    )
    print(f"Training trials assembled: {len(training_dat)}\n")


    print(f"Preparing validation trials ...")
    validation_dat = assemble_trials(
        neural_trials,
        validation_idx,
        time_steps=data["time_steps"],
        step_size=data["step_size"],
        min_lpc_feat=min_lpc_feat,
        max_lpc_feat=max_lpc_feat,
    )
    print(f"Validation trials assembled: {len(validation_dat)}\n")


    return training_dat, validation_dat, sessions, session_layers

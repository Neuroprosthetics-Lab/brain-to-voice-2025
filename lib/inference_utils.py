# ==================================================================================
# Inference utilities for brain-to-voice
# Maitreyee Wairagkar, 2025
# ==================================================================================

import os
import time

from keras.models import load_model, Model
import librosa
import numpy as np
import scipy.io

from lib import data_utils, speech_processor


def load_neural_data(
    neural_data_path: str,
    session: str,
    block: str = '',
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load neural data and cue sentences for a single inference session

    Args:
        neural_data_path : path to the directory containing session folders
        session          : session folder name string
        block            : optional block identifier string
    
    Returns:
        session_name : session name string extracted from the mat file
        sentences    : array of cue sentence strings, one per trial
        spike_pow    : array of spikepow trial arrays
        thresh_cross : array of threshold crossing trial arrays
    """

    if block == '':
        neural_block_path = os.path.join(neural_data_path, session, f"{session}_neural_data")
    else:
        neural_block_path = os.path.join(neural_data_path, block)

    neural_dat = scipy.io.loadmat(neural_block_path, squeeze_me=True)
    print(f"Loaded keys: {list(neural_dat.keys())}")

    session_name = neural_dat["session"]
    spike_pow = np.squeeze(neural_dat["spikepow_trials"])
    thresh_cross = np.squeeze(neural_dat["threshcross_trials"])
    if "cues" in neural_dat.keys():
        sentences = np.squeeze(neural_dat["cues"])
    else:
        sentences = np.squeeze(neural_dat["sentences"])

    n_trials = len(sentences)
    print(f"Session: {session_name} | Trials: {n_trials}")

    return session_name, sentences, spike_pow, thresh_cross


def load_b2voice_model(model_full_path: str) -> Model:
    """
    Load a pre-trained brain-to-voice inference model from disk.

    Args:
        model_full_path : full path to .h5 inference model

    Returns:
        b2voice_model : loaded Keras model ready for inference
    """
    print(f"Loading model from: {model_full_path}")

    b2voice_model = load_model(model_full_path)
    b2voice_model.summary()

    return b2voice_model


def run_inference_single_trial(
    b2voice_model: Model,
    spike_pow: np.ndarray,
    thresh_cross: np.ndarray,
    seq_len: int,
    min_lpc_feat: np.ndarray,
    max_lpc_feat: np.ndarray,
) -> np.ndarray:
    """
    Run brain-to-voice inference on a single trial.
    Concatenates spikepow and threshcross features, extracts sliding windows,
    and returns the raw predicted LPCNet features after scaling back to 
    original range.

    Args:
        b2voice_model : loaded Keras inference model
        spike_pow     : spikepow feature array for this trial of shape (T, C//2)
        thresh_cross  : threshold crossing feature array for this trial of shape (T, C//2)
        seq_len       : sliding window length in bins
        min_lpc_feat  : per-feature minimum values used during training normalisation
        max_lpc_feat  : per-feature maximum values used during training normalisation

    Returns:
        prediction : predicted LPCNet features of shape (T, n_features) in normalised range
    """
    # Get neural features for this trial
    neural_features = np.concatenate([spike_pow, thresh_cross], axis=1)

    # Divide the features into sliding windows of the required squence length
    feat_mat = np.squeeze(
        data_utils.get_sliding_windows(neural_features, window_len=seq_len, step_size=1)
    )

    # Perform the trial-wise inference
    prediction = b2voice_model.predict(feat_mat)
    print(f"Predicted LPCNet features shape: {prediction.shape}")

    # Scale predictions back to original LPCNet feature range
    prediction = data_utils.scale_to_original_range(prediction, min_lpc_feat, max_lpc_feat)

    return prediction


def save_lpcnet_features(
    prediction: np.ndarray,
    trial: int,
    session_name: str,
    aud_dat_path: str,
):
    """
    Save predicted LPCNet features for a single trial as a .mat file 
    in the audio output folder

    Args:
        prediction   : predicted LPCNet feature array in normalised range of shape (T, n_features)
        trial        : trial index used for labelling the output filename
        session_name : session name string used in the output filename
        aud_dat_path : directory path where the .mat file is saved
    """
    save_path = os.path.join(aud_dat_path, f"lpcnet_features_{session_name}_tr{trial}.mat")

    scipy.io.savemat(
        save_path,
        {
            "predicted_lpcnet_features": prediction,
        },
    )
    print(f"LPCNet features saved to: {save_path}")


def synthesize_speech(
    prediction: np.ndarray,
    pred_fname: str,
    lpcnet_path: str,
    aud_dat_path: str,
    ffmpeg_path: str= "ffmpeg"
) -> tuple[np.ndarray, int]:
    """
    Reconstruct full LPCNet features, and synthesize a wav file using 
    pre-trained LPCNet decoding

    Args:
        prediction   : predicted LPCNet features in normalised range of shape (T, n_features)
        pred_fname   : base filename (without extension) for the output wav file
        lpcnet_path  : path to the LPCNet directory containing the lpcnet_demo executable
        aud_dat_path : path to directory where synthesized audio files are saved
        ffmpeg_path  : path to the ffmpeg executable

    Returns:
        pred_sig : synthesized audio signal as a 1D float array
        fs_pred  : sampling frequency of the synthesized audio in Hz
    """
    # Reconstruct full 36-dim LPCNet features from predicted cepstrum and pitch
    features_to_decode = speech_processor.reconstruct_lpcnet_features(prediction.copy())

    # Decode to wav using LPCNet
    speech_processor.decode_lpcnet_features_to_wav(
        features_to_decode, pred_fname, lpcnet_path, aud_dat_path, ffmpeg_path
    )

    # Wait for LPCNet synthesis to finish
    time.sleep(1)

    # Load the synthesized audio
    wav_path = os.path.join(aud_dat_path, f"{pred_fname}.wav")
    pred_sig, fs_pred = librosa.load(wav_path, sr=None)

    return pred_sig, fs_pred


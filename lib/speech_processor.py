# ==================================================================================
# Speech synthesis and vocoder functions for the offline implementation of brain-to-voice synthesis
# Maitreyee Wairagkar, 2025
# ==================================================================================

import math
import os
import platform
import shutil
import subprocess
import time

import librosa
import numpy as np
import soundfile

# constants
_N_CEPS = 18  # fixed number of cepstral coefficients for LPCNet
_N_PITCH = 2  # fixed number of pitch components for LPCNet
_N_LPC = 16   # fixed number of LPC coefficients for LPCNet
 
def encode_lpcnet_features_from_wav(
    wav_filename: str,
    lpcnet_path: str,
    temp_files_path: str,
    ffmpeg_path: str = "ffmpeg",
    save_features_bin: bool = False,
) -> np.ndarray:
    """
    Encode LPCNet features from a wav file using the LPCNet encoder

    Args:
        wav_filename      : path to the input wav file
        lpcnet_path       : path to the LPCNet directory containing the lpcnet_demo executable
        temp_files_path   : path to directory for storing temporary intermediate files
        ffmpeg_path       : path to the ffmpeg executable
        save_features_bin : whether to save the encoded features as a binary .f32 file

    Returns:
        feature_vectors : LPCNet feature array of shape (n_frames [10ms], 36)
    """
    temp_wav_filename = os.path.join(temp_files_path, "temp_wav.wav")
    input_pcm_filename = os.path.join(temp_files_path, "input.pcm")
    features_f32_filename = os.path.join(temp_files_path, "features.f32")
    saved_features_32_filename = os.path.join(
        temp_files_path, "features", f"features_{os.path.splitext(os.path.basename(wav_filename))[0]}.f32"
    )

    # Delete files from previous run to be safe
    delete_temp_files([temp_wav_filename, input_pcm_filename, features_f32_filename])

    # Load the wav audio
    sampling_rate = librosa.get_samplerate(wav_filename)
    sig, fs = librosa.load(wav_filename, sr=sampling_rate)
    soundfile.write(temp_wav_filename, sig, fs, subtype="PCM_24")

    # Strip LD_LIBRARY_PATH to avoid conda lib conflicts with system ffmpeg
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)

    # Convert wav to pcm at 16kHz required for LPCNet encoding
    out = 1
    if platform.system() == "Darwin":  # Mac
        while out != 0:
            result = subprocess.run(
                "echo y | {} -i {} -f s16le -ar 16000 -acodec pcm_s16le {} > /dev/null 2>&1".format(
                    ffmpeg_path, temp_wav_filename, input_pcm_filename
                ),
                shell=True,
                env=env,
            )
            out = result.returncode

    if platform.system() == "Linux":
        while out != 0:
            result = subprocess.run(
                "yes 2>/dev/null | {} -i {} -f s16le -ar 16000 -acodec pcm_s16le {} 2>/dev/null".format(
                    ffmpeg_path, temp_wav_filename, input_pcm_filename
                ),
                shell=True,
                env=env,
            )
            out = result.returncode

    # Encode pcm to LPCNet features using lpcnet_demo
    out = 1
    while out != 0:
        result = subprocess.run(
            "{}/lpcnet_demo -features {} {}".format(
                lpcnet_path, input_pcm_filename, features_f32_filename
            ),
            shell=True,
            env=env,
        )
        out = result.returncode

    print(f"LPCNet feature encoding exit code: {out}")

    # Save the features file if asked
    if save_features_bin:
        time.sleep(2)
        shutil.copy(features_f32_filename, saved_features_32_filename)
        time.sleep(2)

    # Load and convert raw bytes to float32 - 36 features per 10ms frame
    float32 = np.fromfile(features_f32_filename, dtype=np.float32)
    n_features = 36  # constant for LPCNet, cannot be changed
    feature_vectors = np.reshape(
        float32, (int(float32.shape[0] / n_features), n_features)
    )

    return feature_vectors


def decode_lpcnet_features_to_wav(
    feature_vectors: np.ndarray,
    output_filename: str,
    lpcnet_path: str,
    temp_files_path: str,
    ffmpeg_path: str = "ffmpeg"
) -> str:
    """
    Decode LPCNet feature vectors to a wav file using the LPCNet synthesiser

    Args:
        feature_vectors : LPCNet feature array to decode
        output_filename : base name for the output PCM and wav files
        lpcnet_path     : path to the LPCNet directory containing the lpcnet_demo executable
        temp_files_path : path to directory for storing temporary intermediate files
        ffmpeg_path     : path to the ffmpeg executable

    Returns:
        output_wav_filename : path to the decoded output wav file
    """

    features_f32_filename = os.path.join(temp_files_path, "predicted_features.f32")
    output_pcm_filename = os.path.join(temp_files_path, f"{output_filename}.pcm")
    output_wav_filename = os.path.join(temp_files_path, f"{output_filename}.wav")

    feature_vectors = np.reshape(feature_vectors, (feature_vectors.size,))

    # Write LPCNet features as a binary file
    feature_vectors.tofile(features_f32_filename)

    print(f"Features shape: {feature_vectors.shape}")

    # Decode features from bin file to pcm using LPCNet
    os.system(
        lpcnet_path
        + "/lpcnet_demo -synthesis {} {}".format(
            features_f32_filename, output_pcm_filename
        )
    )

    # Convert the decoded PCM to wav
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    subprocess.run(
        "yes 2>/dev/null | {} -f s16le -ar 16000 -ac 1 -i {} {} 2>/dev/null".format(
            ffmpeg_path, output_pcm_filename, output_wav_filename
        ),
        shell=True,
        env=env,
    )

    return output_wav_filename


def reconstruct_lpcnet_features(cepstrum_and_pitch: np.ndarray) -> np.ndarray:
    """
    Reconstruct full LPCNet features from reduced cepstrum and pitch features.
    LPC coefficients are estimated from cepstral coefficients and concatenated
    with cepstrum and pitch to form the complete LPCNet feature vector.

    Args:
    cepstrum_and_pitch : array of cepstral coefficients and pitch features
                         of shape (n_frames [10ms], n_cepstral_coeff + 2)

    Returns:
        reconstructed_lpcnet_features : full LPCNet feature array of shape (n_frames [10ms], 36)
                                        containing cepstrum, pitch, and reconstructed LPC coefficients
    """

    # Reconstruct the LPC coefficients from cepstrum
    n_ceps = cepstrum_and_pitch[:, :-_N_PITCH].shape[1]  # extract number of cepstral coefficients

    lpc_predicted = np.zeros((cepstrum_and_pitch.shape[0], _N_LPC), dtype="float32")

    # Add zeros if reduced dimensions of cepstral coefficients are used for reconstruction of LPC
    zero_added_cepstrum = np.zeros((cepstrum_and_pitch.shape[0], _N_CEPS), dtype="float32")
    zero_added_cepstrum[:, :n_ceps] = cepstrum_and_pitch[:, :-_N_PITCH]  # extract and add cepstrum coefficients

    # Estimate LPC from cepstral coefficients
    for t in range(cepstrum_and_pitch.shape[0]):
        lpc_predicted[t, :] = cepstrum_to_lpc(zero_added_cepstrum[t, :])

    # Reconstruct all features required for LPCNet decoding
    reconstructed_lpcnet_features = np.concatenate(
        [
            zero_added_cepstrum,  # add cepstrum coefficients
            cepstrum_and_pitch[:, -_N_PITCH:],  # add pitch features (last two features are always pitch features in LPCNet)
            lpc_predicted,
        ],  # add reconstructed lpc coefficients
        axis=1,
        dtype="float32",
    )

    return reconstructed_lpcnet_features


def lpcnet_features_dimensionality_reduction(
    all_lpcnet_features: np.ndarray,
    n_cepstral_coeff: int = _N_CEPS,
) -> np.ndarray:
    """
    Reduce LPCNet feature dimensionality by retaining only the first n cepstral
    coefficients and 2 pitch features, discarding the 16 LPC coefficients.
    LPC coefficients can later be reconstructed from the cepstral coefficients.

    Args:
        all_lpcnet_features : full LPCNet feature array of shape (n_frames [10ms], 36)
        n_cepstral_coeff    : number of cepstral coefficients to retain.
                              Dimensionality reduction occurs if less than 18.
                              Default is 18 (no reduction).

    Returns:
        dim_reduced_features : reduced feature array of shape (n_frames [10ms], n_cepstral_coeff + 2)
    """

    cepstral_coeff = all_lpcnet_features[:, :n_cepstral_coeff]
    pitch_features = all_lpcnet_features[:, _N_CEPS:_N_CEPS +_N_PITCH]

    dim_reduced_features = np.concatenate([cepstral_coeff, pitch_features], axis=1)

    return dim_reduced_features


def cepstrum_to_lpc(cepstral_coeff: np.ndarray) -> np.ndarray:
    """
    Compute LPC coefficients from cepstral coefficients.
    Reference: https://github.com/xiph/LPCNet/issues/93
    This follows the function lpc_from_cepstrum from freq.c from LPCNet C source code
    All values are kept in float32 to avoid precision errors.

    Args:
        cepstral_coeff : cepstral coefficients for a single frame of size NB_BANDS (18,)

    Returns:
        lpc_ : LPC coefficients of shape (LPC_ORDER,) i.e. (16,)
    """
    # input is cepstrum of single frame of size NB_BANDS
    cepstral_coeff = np.float32(cepstral_coeff)

    # ********   LPCNet constants (used only in this function - not tunable ***********************
    # Note: these are hardcoded constants, the LPCNet output will not be valid with any other values
    FRAME_SIZE_5MS = 2
    OVERLAP_SIZE_5MS = 2
    WINDOW_SIZE_5MS = FRAME_SIZE_5MS + OVERLAP_SIZE_5MS
    FRAME_SIZE = 80 * FRAME_SIZE_5MS
    OVERLAP_SIZE = 80 * OVERLAP_SIZE_5MS

    WINDOW_SIZE = FRAME_SIZE + OVERLAP_SIZE
    FREQ_SIZE = WINDOW_SIZE // 2 + 1
    NB_BANDS = cepstral_coeff.size  # 18
    LPC_ORDER = 16

    compensation = np.array(
        [
            0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.666667, 0.5, 0.5, 0.5, 0.333333, 0.25, 0.25, 0.2, 0.166667, 0.173913,
        ],
        dtype="float32",
    )

    eband5ms = [
        # 0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k*/
          0, 1,  2,  3,  4,   5, 6,  7,  8,   10,12, 14, 16,  20,24, 28, 34,  40,
    ]

    # Values required for inverse discrete transform
    dct_table = np.zeros(NB_BANDS * NB_BANDS, dtype="float32")
    for i in range(NB_BANDS):
        for j in range(NB_BANDS):
            dct_table[i * NB_BANDS + j] = np.cos((i + 0.5) * j * math.pi / NB_BANDS)
            if j == 0:
                dct_table[i * NB_BANDS + j] *= np.sqrt(0.5)
    dct_table = np.float32(dct_table)

    # ******** LPCNet local functions *******************

    # Function for inverse discrete transform
    def idct_(in_):
        """
        Inverse discrete cosine transform for a single frame using the precomputed DCT table.

        Args:
            in_ : input cepstral coefficients of shape (NB_BANDS,)

        Returns:
            out : reconstructed band energies of shape (NB_BANDS,) in float32
        """
        out = np.zeros(NB_BANDS, dtype="float32")
        for i in range(NB_BANDS):
            sum = 0
            for j in range(NB_BANDS):
                sum += in_[j] * dct_table[i * NB_BANDS + j]
            out[i] = sum * np.sqrt(2.0 / NB_BANDS)
        return np.float32(out)

    def interp_band_gain(bandE):
        """
        Interpolate band gains from Bark-scale bands to linear frequency bins.

        Args:
        bandE : Bark-scale band energy array of shape (NB_BANDS,)

        Returns:
            g : interpolated linear frequency gains of shape (FREQ_SIZE,) in float32
        """
        g = np.zeros(FREQ_SIZE, dtype="float32")
        for i in range(NB_BANDS - 1):
            band_size = (eband5ms[i + 1] - eband5ms[i]) * WINDOW_SIZE_5MS
            for j in range(band_size):
                frac = np.float32(j / band_size)
                g[(eband5ms[i] * WINDOW_SIZE_5MS) + j] = np.float32(
                    (1 - frac) * bandE[i] + frac * bandE[i + 1]
                )
        return np.float32(g)

    def _lpcnet_lpc(ac, p):
        """
        Compute LPC coefficients from autocorrelation using the Levinson-Durbin algorithm.

        Args:
            ac : autocorrelation array of shape (p+1,)
            p  : LPC order

        Returns:
            error : prediction error energy
            lpc   : LPC coefficients of shape (p,)
            rc    : reflection coefficients of shape (p,)
        """
        error = ac[0]
        lpc = np.zeros(p)
        rc = np.zeros(p).astype(float)
        if ac[0] != 0:
            for i in range(p):
                rr = 0
                for j in range(i):
                    rr += lpc[j] * ac[i - j]
                rr += ac[i + 1]
                r = -rr / error
                rc[i] = r
                lpc[i] = r
                for j in range(int((i + 1) / 2)):
                    tmp1 = lpc[j]
                    tmp2 = lpc[i - 1 - j]
                    lpc[j] = tmp1 + (r * tmp2)
                    lpc[i - 1 - j] = tmp2 + (r * tmp1)
                error -= (r * r) * error
                if error < (ac[0] / (2**10)):
                    break
                if error < (0.001 * ac[0]):
                    break
        return error, lpc, rc

    # *********** Implement Cepstrum to LPC **************

    # Step 1: Get Inverse Descrete Transform TODO: check if copy is necessary here
    tmp = cepstral_coeff[:NB_BANDS].copy()
    tmp[0] += 4
    Ex = idct_(tmp)  # inverse cepstrums to Bank-scale spectrogram

    # Step 2: Get Ex Bark-scale spectrogram
    Ex = (10.0**Ex) * compensation[:NB_BANDS]
    Ex = np.float32(Ex)

    # Step 3: Get LPC from spectral bands - interpolate band gain
    Xr = interp_band_gain(Ex)  #  interpolate linear spectrogram

    # Step 4: Get autocorrelation from linear spectrum (done in inverse_transform function in freq.c)
    acr = np.real(np.fft.irfft(Xr))  #  calculate autocorrelation
    acr = acr[: LPC_ORDER + 1]
    acr[0] += acr[0] * 0.0001 + 320 / 12 / 38.0
    for i in range(1, LPC_ORDER + 1):
        acr[i] *= 1 - 0.00006 * i * i
    acr = np.float32(acr)

    # Step 5: get LPC from autocorrelation
    e, lpc_, rc = _lpcnet_lpc(
        acr, LPC_ORDER
    )  #  calculate lpc from autocorrelation using Levinson Dublin algorithm
    lpc_ = np.float32(lpc_)

    return lpc_


def delete_temp_files(filelist: list[str]):
    """
    Delete temporary files if they exist

    Args:
        filelist : list of file paths to delete
    """

    for file in filelist:
        if os.path.exists(file):
            os.remove(file)

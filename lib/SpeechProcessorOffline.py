# Speech processing functions for offline implementation of brain-to-voice synthesis

# Author: Maitreyee Wairagkar, 2025
# Created: 2022-06-08
# Last update: 2025-04-30

import numpy as np
import math 
import random
from matplotlib import pyplot as plt

import librosa
import librosa.display
import soundfile

import time  
import os
import platform
import shutil


def encode_lpcnet_features_from_wav(wav_filename, lpcnet_path, temp_files_path, save_features_bin = False, length=1):
    
    temp_wav_filename  = temp_files_path+'/temp_wav.wav'
    input_pcm_filename = temp_files_path+'/input.pcm'
    features_f32_filename  = temp_files_path+'/features.f32' 
    saved_features_32_filename = temp_files_path+'/features/features_'+wav_filename[51:-4]+'.f32' # for emg data TODO - use regex
    #saved_features_32_filename = temp_files_path+'/features/features_'+wav_filename[-15:-4]+'.f32' #wav_filename[-15:-4] gives file name without .wav and without full path

    # delete the above files if they exist from previous run to be safe
    delete_temp_files([temp_wav_filename, input_pcm_filename, features_f32_filename])

    # I. stretch the signal 
    # Get sampling frequency
    sampling_rate = librosa.get_samplerate(wav_filename)

    # Load the signal
    sig, Fs = librosa.load(wav_filename, sr=sampling_rate)
    
    # stretch to match the length of neural data
    #sig1 = stretch_to_match_length(sig1, length, Fs, N_FFT)
  
    # write stretched audio to a temporary file
    soundfile.write(temp_wav_filename, sig, Fs, subtype='PCM_24')

    # II. Encode LPCNet features from the stretched wav file
    # Convert the wav file to pcm with Fs=16k required for LPCNet encoding using ffmpeg by running the commandline command
    # send argument echo y before ffmpeg command to auto answer yes to the prompt for overwriting the file

    out = []
    if platform.system() == 'Darwin': # Mac
        while out != 0:
            out = os.system('Echo y| ffmpeg -i {} -f s16le -ar 16000 -acodec pcm_s16le {}  &> /dev/null &'.format(temp_wav_filename, input_pcm_filename)) # &> /dev/null & is to hide the stdout
    if platform.system() == 'Linux':
        while out != 0:
            out = os.system('yes 2>/dev/null | ffmpeg -i {} -f s16le -ar 16000 -acodec pcm_s16le {}  2>/dev/null'.format(temp_wav_filename, input_pcm_filename)) # &> /dev/null & is to hide the stdout
  
    # Encode the above pcm file using LPCNet to get features that will be used for trainig the model
    # copy lpcnet_demo executable file in the same folder as this script
    out = []
    while out != 0: # out = 0 = successful feature encoding
        out = os.system(lpcnet_path+'/lpcnet_demo -features {} {}'.format(input_pcm_filename,features_f32_filename))
    print(out)

    # Save the features file if asked
    if save_features_bin == True:
        time.sleep(2)
        shutil.copy(features_f32_filename, saved_features_32_filename)
        time.sleep(2)
    
    # Load and convert raw bytes to float 32. There are 36 features per 10ms frame
    float32 = np.fromfile(features_f32_filename,  dtype=np.float32)
    
    num_features = 36

    feature_vectors = np.reshape(float32, (int(float32.shape[0]/num_features), num_features))
    
    return feature_vectors


def decode_lpcnet_features_to_wav(feature_vectors, output_filename, lpcnet_path, temp_files_path):
    
    features_f32_filename = temp_files_path+'/predicted_features.f32'
    output_pcm_filename = temp_files_path+ '/'+output_filename+'.pcm'
    output_wav_filename = temp_files_path+ '/'+output_filename+'.wav'

    feature_vectors = np.reshape(feature_vectors, (feature_vectors.size,))
    
    # Write LPCNet features as a binary file 
    feature_vectors.tofile(features_f32_filename)
    
    print(feature_vectors.shape)
    '''
    plt.figure()
    plt.plot(feature_vectors)
    plt.show()
    '''
    # Decode features from bin file to pcm using LPCNet
    os.system(lpcnet_path+'/lpcnet_demo -synthesis {} {}'.format(features_f32_filename, output_pcm_filename))

    # Convert the decoded PCM to wav
    os.system('Echo y| ffmpeg -f s16le -ar 16000 -ac 1 -i {} {} &> /dev/null &'.format(output_pcm_filename, output_wav_filename))
    # &> /dev/null & is to hide the output https://askubuntu.com/questions/150844/how-to-really-hide-terminal-output
    
    return output_wav_filename


def reconstruct_lpcnet_features(cepstrum_and_pitch):
    
    # 1. Reconstruct the LPC coefficients from cepstrum
    n_ceps = cepstrum_and_pitch[:,:-2].shape[1]       # number of cepstral coefficients
    
    lpc_predicted = np.zeros((cepstrum_and_pitch.shape[0], 16), dtype='float32')
    
    # Add zeros if reduced dimensions of cepstral coefficients are used for reconstruction of LPC
    zero_added_cepstrum = np.zeros((cepstrum_and_pitch.shape[0], 18), dtype = 'float32')
    zero_added_cepstrum[:, :n_ceps] = cepstrum_and_pitch[:, :-2]  # add cepstrum coefficients
    
    # Estimate LPC from cepstral coefficients
    for t in range(cepstrum_and_pitch.shape[0]):
        lpc_predicted[t, :] = cepstrum_to_lpc(zero_added_cepstrum[t,:])

        
    # 2. Reconstruct all features required for LPCNet decoding
    reconstructed_lpcnet_features = np.concatenate([zero_added_cepstrum,        # add cepstrum coefficients
                                                    cepstrum_and_pitch[:, -2:], # add pitch features 
                                                    lpc_predicted],             # add reconstructed lpc coefficients
                                                    axis = 1, dtype='float32' )
        
    
    return reconstructed_lpcnet_features


# TODO: add error handling on number of cepstral coeffs
def lpcnet_fetures_dimensionality_reduction(all_lpcnet_features, n_cepstral_coeff=18):
    # LPCNet features: 36 -> 18 cepstral coeff 2 pitch features and 16 LPC coeffs
    # To reduce the dimensionality for training the input, only keep first n cepstral coeffs and 2 pitch features
    # LPC coeff can be later reconstructed from the cepstral_coeff

    cepstral_coeff = all_lpcnet_features[:, :n_cepstral_coeff]
    pitch_features = all_lpcnet_features[:, 18:20]

    dim_reduced_features = np.concatenate([cepstral_coeff, pitch_features], axis=1)

    return dim_reduced_features


# Compute LPC coeff from Cepstrum:
# https://github.com/xiph/LPCNet/issues/93
# This follows the function lpc_from_cepstrum from freq.c from LPCNet C source code
# Important to get all the values in float 32 - beware there can be precision errors
def cepstrum_to_lpc(cepstral_coeff):
    # input is cepstrum of single frame of size NB_BANDS
    cepstral_coeff = np.float32(cepstral_coeff)

    # ********   LPCNet constants ***********************
    FRAME_SIZE_5MS   = 2
    OVERLAP_SIZE_5MS = 2
    WINDOW_SIZE_5MS  = FRAME_SIZE_5MS + OVERLAP_SIZE_5MS
    FRAME_SIZE       = 80 * FRAME_SIZE_5MS
    OVERLAP_SIZE     = 80 * OVERLAP_SIZE_5MS

    WINDOW_SIZE      = FRAME_SIZE + OVERLAP_SIZE
    FREQ_SIZE        = WINDOW_SIZE//2 + 1
    NB_BANDS         = cepstral_coeff.size # 18 *********
    LPC_ORDER        = 16
    
    compensation = np.array([0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.666667, 0.5, 0.5, 
                             0.5, 0.333333, 0.25, 0.25, 0.2, 0.166667, 0.173913], dtype = 'float32')
  
    eband5ms = [
    # 0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k*/
      0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40
    ]
    
    # Values required for inverse descrete transform 
    dct_table = np.zeros(NB_BANDS * NB_BANDS, dtype='float32')
    for i in range(NB_BANDS):
        for j in range(NB_BANDS):
            dct_table[i*NB_BANDS + j] = np.cos((i+.5)*j*math.pi/NB_BANDS)
            if j==0:
                dct_table[i*NB_BANDS + j] *= np.sqrt(.5)
    dct_table = np.float32(dct_table)
    
    # ******** LPCNet local functions *******************
    
    # Function for inverse descrete transform
    def idct_(in_):
        out = np.zeros(NB_BANDS, dtype='float32')
        for i in range(NB_BANDS):
            sum = 0
            for j in range(NB_BANDS):
                sum += in_[j] * dct_table[i*NB_BANDS + j]
            out[i] = sum * np.sqrt(2./NB_BANDS)
        return np.float32(out)
      
    def interp_band_gain(bandE):
        g = np.zeros(FREQ_SIZE, dtype='float32')
        for i in range(NB_BANDS - 1):
            band_size = (eband5ms[i+1]-eband5ms[i])*WINDOW_SIZE_5MS
            for j in range(band_size):
                frac = np.float32(j / band_size)
                g[(eband5ms[i]*WINDOW_SIZE_5MS) + j] = np.float32((1-frac) * bandE[i] + frac * bandE[i+1])
        return np.float32(g)
    
    
    def _lpcnet_lpc(ac, p):
        error = ac[0]
        lpc = np.zeros(p)
        rc = np.zeros(p).astype(float)
        if ac[0] != 0:
            for i in range(p):
                rr = 0
                for j in range(i):
                    rr += lpc[j] * ac[i - j]
                rr += (ac[i + 1])
                r = -rr / error
                rc[i] = r
                lpc[i] = (r)
                for j in range(int((i+1) / 2)):
                    tmp1 = lpc[j]
                    tmp2 = lpc[i-1-j]
                    lpc[j]     = tmp1 + (r * tmp2)
                    lpc[i-1-j] = tmp2 + (r * tmp1)
                error -= ((r * r) * error)
                if error < (ac[0]  / (2**10)):
                    break
                if error < (0.001 * ac[0]):
                    break
        #print(f'ac[0]={ac[0]} error={error}')
        return error, lpc, rc

        
    # *********** Implement Cepstrum to LPC **************
    
    # Step 1: Get Inverse Descrete Transform TODO: check if copy is necessary here 
    tmp = cepstral_coeff[:NB_BANDS].copy()
    tmp[0] += 4
    Ex = idct_(tmp) # inverse cepstrums to Bank-scale spectrogram

    # Step 2: Get Ex Bark-scale spectrogram
    Ex = (10.0 ** Ex) * compensation[:NB_BANDS]
    Ex = np.float32(Ex)
    
    # Step 3: Get LPC from spectral bands - interpolate band gain
    Xr = interp_band_gain(Ex) #  interpolate linear spectrogram 
    
    # Step 4: Get autocorrelation from linear spectrum (done in inverse_transform function in freq.c)
    acr = np.real(np.fft.irfft(Xr)) #  calculate autocorrelation
    acr = acr[:LPC_ORDER+1]
    acr[0] += acr[0] * 0.0001 + 320/12/38.
    for i in range(1, LPC_ORDER+1):
        acr[i] *= (1 - 0.00006 * i * i)
    acr = np.float32(acr)   
        
    # Step 5: get LPC from autocorrelation
    e, lpc_, rc = _lpcnet_lpc(acr, LPC_ORDER) #  calculate lpc from autocorrelation using Levinson Dublin algorithm
    lpc_ = np.float32(lpc_)
    
    return lpc_


def delete_temp_files(filelist):

    for file in filelist:
        if os.path.exists(file):
            os.remove(file)
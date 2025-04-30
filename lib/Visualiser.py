# Functions to visualise neural and voice data 

# Author: Maitreyee Wairagkar, 2025
# Created: 2022-06-09
# Last update: 2023-09-14

import numpy as np
from matplotlib import pyplot as plt

import librosa
import librosa.display


# Print  spectrogram
def print_spectrogram(spectrogram, Fs, n_fft, step_size, fmin=50, fmax = 7000):
    
    plt.figure()

    librosa.display.specshow(np.transpose(spectrogram), sr=Fs, n_fft=n_fft, hop_length=step_size, y_axis='mel', x_axis='time', 
                             fmin = fmin, fmax = fmax)
    plt.colorbar()
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Mel-Spectrogram with log amplitude')

    plt.show()


def show_img(dat, d_subplots=[1,1], title='', xlabel='', ylabel='', supxlabel = '', supylabel = '', 
             suptitle = '', figsize=[5,5], cmap='jet', fontsize=22, cbar = False, cbar_ori='vertical'):
    '''
    Plot 2D images in subplots
    Args: 
        dat       : 2D array, a list of 2D arrays or a 3D array (which will be plotted as a series of 2D arrays)
        d_subplots: 2D list of subplots dimension [nrows, ncols] 
        title     : list of titles or single title for a single plot
        xlabel    : list of xlabels or single xlabel for a single plot
        ylabel    : list of ylabels or single ylabel for a single plot
        supxlabel : shared xlabel for entire figure
        supylabel : shared ylabel for entire figgure
        suptitle  : shared title for entire figure
    '''
    
    xlabel = list(xlabel)
    ylabel = list(ylabel)
    title  = list(title)
    
    if type(dat) != list:
        if len(dat.shape) == 2: #If a single 2D array is given, add an axix
            dat = dat[np.newaxis, :]

    fig, axs = plt.subplots(d_subplots[0], d_subplots[1], figsize=figsize) 
    
    axs = np.array(axs)
    
    for i, ax in enumerate(axs.reshape(-1)):
        im = ax.imshow(dat[i], aspect='auto', cmap=cmap, interpolation='none')
        if cbar:
            fig.colorbar(im, ax=ax, orientation = cbar_ori)
        
        if len(xlabel)>i:
            ax.set_xlabel(xlabel[i])
        if len(ylabel)>i:
            ax.set_ylabel(xlabel[i])
        if len(title)>i:
            ax.set_title(title[i], fontsize=fontsize)

    fig.suptitle(suptitle, fontsize=fontsize)
    fig.supxlabel(supxlabel, fontsize=fontsize)
    fig.supylabel(supylabel, fontsize=fontsize)
    plt.tight_layout()
    plt.show()

    return plt


def plot_timeseries(dat, title='', xlabel='', ylabel='', color='#1f77b4', figsize=[5,5]):
    # plot columns of a matrix as a separate timeseries
    
    plt.figure(figsize = figsize)
    
    if np.ndim(dat) == 1:
        plt.plot(dat)

    if np.ndim(dat) == 2:
        for c in range(dat.shape[1]):
            plt.plot(dat[:,c])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

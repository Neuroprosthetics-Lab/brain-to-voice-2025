# ==================================================================================
# Functions to visualise neural and voice data
# Maitreyee Wairagkar, 2025
# ==================================================================================

import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np


def plot_spectrogram(
    spectrogram: np.ndarray,
    fs: int,
    n_fft: int,
    step_size: int,
    fmin: float = 50,
    fmax: float = 7000,
):
    """
    Plot a mel-spectrogram with log amplitude using librosa

    Args:
        spectrogram : 2D array of mel-spectrogram values of shape (time, mel_bins)
        fs          : sampling frequency in Hz
        n_fft       : FFT window size
        step_size   : hop length in samples between frames
        fmin        : minimum frequency for mel filterbank in Hz
        fmax        : maximum frequency for mel filterbank in Hz
    """

    plt.figure()

    librosa.display.specshow(
        np.transpose(spectrogram),
        sr=fs,
        n_fft=n_fft,
        hop_length=step_size,
        y_axis="mel",
        x_axis="time",
        fmin=fmin,
        fmax=fmax,
    )
    plt.colorbar()
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Mel-Spectrogram with log amplitude")

    plt.show()


def show_img(
    dat: np.ndarray,
    d_subplots: list[int] = [1, 1],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    supxlabel: str = "",
    supylabel: str = "",
    suptitle: str = "",
    figsize: list[int] = [5, 5],
    cmap: str = "jet",
    fontsize: int = 22,
    cbar: bool = False,
    cbar_ori: str = "vertical",
):
    """
    Plot 2D images in subplots

    Args:
        dat        : 2D array, list of 2D arrays, or 3D array plotted as a series of 2D arrays
        d_subplots : subplot grid dimensions as [nrows, ncols]
        title      : list of titles or a single title string for all subplots
        xlabel     : list of x-axis labels or a single label for all subplots
        ylabel     : list of y-axis labels or a single label for all subplots
        supxlabel  : shared x-axis label for the entire figure
        supylabel  : shared y-axis label for the entire figure
        suptitle   : shared title for the entire figure
        figsize    : figure size as [width, height] in inches
        cmap       : colormap name to use for imshow
        fontsize   : font size for titles and shared labels
        cbar       : whether to display a colorbar for each subplot
        cbar_ori   : colorbar orientation, either 'vertical' or 'horizontal'
    """

    xlabel = list(xlabel)
    ylabel = list(ylabel)
    title = list(title)

    if type(dat) != list:
        if len(dat.shape) == 2:  # If a single 2D array is given, add an axix
            dat = dat[np.newaxis, :]

    fig, axs = plt.subplots(d_subplots[0], d_subplots[1], figsize=figsize)

    axs = np.array(axs)

    for n, ax in enumerate(axs.reshape(-1)):
        im = ax.imshow(dat[n], aspect="auto", cmap=cmap, interpolation="none")
        if cbar:
            fig.colorbar(im, ax=ax, orientation=cbar_ori)

        if len(xlabel) > n:
            ax.set_xlabel(xlabel[n])
        if len(ylabel) > n:
            ax.set_ylabel(ylabel[n])
        if len(title) > n:
            ax.set_title(title[n], fontsize=fontsize)

    fig.suptitle(suptitle, fontsize=fontsize)
    fig.supxlabel(supxlabel, fontsize=fontsize)
    fig.supylabel(supylabel, fontsize=fontsize)
    plt.tight_layout()
    plt.show()

    return plt


def plot_timeseries(
    dat: np.ndarray,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    color: str = "#1f77b4",
    figsize: list[int] = [5, 5],
):
    """
    Plot columns of a matrix as separate time series on a single figure

    Args:
        dat     : 1D array for a single time series or 2D array where each column is plotted separately
        title   : plot title
        xlabel  : x-axis label
        ylabel  : y-axis label
        color   : line color as a hex string
        figsize : figure size as [width, height] in inches
    """

    plt.figure(figsize=figsize)

    if np.ndim(dat) == 1:
        plt.plot(dat)

    elif np.ndim(dat) == 2:
        for c in range(dat.shape[1]):
            plt.plot(dat[:, c])
    else:
        raise ValueError("Input data must be 1D or 2D array")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

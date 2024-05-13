import os
from itertools import pairwise
from typing import (
    Callable,
    List,
    Tuple
)
from zipfile import ZipFile
from datetime import datetime
from multiprocessing.pool import ThreadPool
from tqdm.notebook import tqdm

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from scipy.stats import entropy
from scipy.signal import (
    find_peaks,
    windows,
    welch,
    iirfilter,
    lfilter,
    decimate
)
from scipy.fft import rfft
from scipy.interpolate import interp1d
from tsfel import feature_extraction as ft
import pywt


from vibrodiagnostics import mafaulda


def energy(x: np.array) -> float:
    """Calculate energy of the signal

    :param x: input signal

    :return: energy of the signal
    """
    return np.sum(x ** 2)


def negentropy(x: np.array) -> float:
    """Calculate negentropy of the signal

    :param x: input signal

    :return: negentropy of the signal
    """
    if len(x) == 0:
        return np.nan
    return -entropy((x ** 2) / np.mean(x ** 2))


def signal_to_noise(x: np.array) -> float:
    """Calculate estimated signal to noise ratio (noisiness) of the signal.
    Formula is taken from: https://www.geeksforgeeks.org/signal-to-noise-ratio-formula/

    :param x: input signal

    :return: SNR of the signal
    """
    m = np.mean(x)
    sd = np.std(x)
    return np.where(sd == 0, 0, m / sd)


def spectral_roll_off_frequency(f: np.array, pxx: np.array, percentage: float) -> float:
    """Calculate roll-off frequency. Cumulative sum of energy in spectral bins
    below roll-off frequency is percentage of total energy

    :param f: array of frequency bins
    :param pxx: array of amplitudes at frequency bins
    :param percentage: ratio of total energy below the roll-off frequency

    :return: roll-off frequency in Hz
    """
    return f[np.argmax(np.cumsum(pxx**2) >= percentage * energy(pxx))]


def temporal_variation(x: pd.Series, window: int) -> List[float]:
    """Calculate temporal variations in successive frequency spectra. It is a
    inverse correlation of pairs formed from overlapping windows.

    :param x: input signal in time domain
    :param window: length of Hann FFT window

    :return: array of temporal variations
    """
    overlap = 0.5
    step = int(window * overlap)
    v = x.to_numpy()
    spectra = [
        np.absolute(rfft(v[i:i+window] * windows.hann(window)))
        for i in range(0, len(v) - window, step)
    ]
    fluxes = [
        1 - np.corrcoef(psd1, psd2) for psd1, psd2 in pairwise(spectra)
    ]
    return fluxes


def envelope_signal(f: np.array, pxx: np.array) -> np.array:
    """Approximate envelope of frequency spectrum with quadratic interpolation
    between peaks

    :param f: array of frequency bins
    :param pxx: array of amplitudes at frequency bins

    :return: envelope of the amplitudes
    """
    # peaks = mms_peak_finder(pxx)
    peaks, _ = find_peaks(pxx)
    try:
        envelope = interp1d(f[peaks], pxx[peaks], kind='quadratic', fill_value='extrapolate')
    except ValueError:
        return []

    y_env = envelope(f)
    y_env[y_env < 0] = 0
    return y_env


def spectral_transform(x: pd.Series, window: int, fs: int) -> Tuple[np.array, np.array]:
    """Estimate frequency spectrum using Welch's method. Partition the signal with
    Hann window and 50% overlap.

    :param x: input signal in time domain
    :param window: length of Hann FFT window
    :param fs: sampling frequency in Hz of the input signal

    :return: envelope of the amplitudes
    """
    overlap = 0.5
    step = int(window * overlap)
    v = x.to_numpy()

    f, pxx = welch(
        v,
        fs=fs,
        window='hann',
        nperseg=window,
        noverlap=step,
        scaling='spectrum',
        average='mean',
        detrend='constant',
        return_onesided=True
    )
    return f, pxx


def time_features_calc(
            df: pd.DataFrame,
            col: str,
            fs: int,
            window: int
        ) -> List[Tuple[str, pd.DataFrame]]:
    """Extract complete set of features in time-domain (TD set)

    :param df: data frame with input time series in columns
    :param col: column in data frame to use for extraction
    :param fs: sampling frequency in Hz of the time series
    :param window: length of FFT window (not used)

    :return: list of time-domain features in pairs of feature name
        with column prefix and its value
    """
    x = df[col]
    features = [
        ('zerocross', [ft.zero_cross(x) / len(x)]),
        ('pp', [ft.pk_pk_distance(x)]),
        ('aac', [np.mean(np.absolute(np.diff(x)))]),
        ('rms', [ft.rms(x)]),
        ('skewness', [ft.skewness(x)]),
        ('kurtosis', [ft.kurtosis(x)]),
        ('shape', [ft.rms(x) / np.mean(np.absolute(x))]),
        ('crest', [np.max(np.absolute(x)) / ft.rms(x)]),
        ('impulse', [np.max(np.absolute(x)) / np.mean(np.absolute(x))]),
        ('clearance', [np.max(np.absolute(x)) / (np.mean(np.sqrt(np.absolute(x))) ** 2)]),
    ]
    return [(f'{col}_{f[0]}', f[1]) for f in features]


def frequency_features_calc(
            df: pd.DataFrame,
            col: str,
            fs: int,
            window: int
        ) -> List[Tuple[str, pd.DataFrame]]:
    """Extract complete set of features in frequency-domain (FD set)

    :param df: data frame with input time series in columns
    :param col: column in data frame to use for extraction
    :param fs: sampling frequency in Hz of the time series
    :param window: length of FFT window

    :return: list of frequency-domain features in pairs of feature name
        with column prefix and its value
    """
    f, pxx = spectral_transform(df[col], window, fs)
    fluxes = temporal_variation(df[col], window)
    envelope_spectrum = envelope_signal(f, pxx)

    features = [
        ('centroid', [np.average(f, weights=pxx)]),
        ('std', [ft.calc_std(pxx)]),
        ('skewness', [ft.skewness(pxx)]),
        ('kurtosis', [ft.kurtosis(pxx)]),
        ('roll_on', [spectral_roll_off_frequency(f, pxx, 0.05)]),
        ('roll_off', [spectral_roll_off_frequency(f, pxx, 0.85)]),
        ('flux', [np.mean(fluxes)]),
        ('noisiness', [signal_to_noise(pxx)]),
        ('energy', [energy(pxx)]),
        ('entropy', [entropy(pxx / np.sum(pxx))]),
        ('negentropy', [negentropy(envelope_spectrum)])
    ]
    return [(f'{col}_{f[0]}', f[1]) for f in features]


def wavelet_features_calc(
            df: pd.DataFrame,
            col: str,
            fs: int,
            window: int
        ) -> List[Tuple[str, pd.DataFrame]]:
    """Extract wavelet coefficients for Meyer wavelet and six levels deep.
    Each wavelet coefficient
    produces four features (energy, energy ratio, kurtosis, negentropy)

    :param df: data frame with input time series in columns
    :param col: column in data frame to use for extraction
    :param fs: sampling frequency in Hz of the time series
    :param window: length of FFT window  (not used)

    :return: list of wavelet-domain features in pairs of feature name
        with column prefix and its value
    """

    max_level = 6
    wavelet = 'dmey'
    ordering = 'freq'
    x = df[col]
    result = []
    columns = mafaulda.BEARINGS_COLUMNS

    for col in columns:
        wp = pywt.WaveletPacket(data=x, wavelet=wavelet, mode='symmetric')

        for feature in ('energy', 'energy_ratio', 'kurtosis', 'negentropy'):
            wpd_header = []
            row = {
                'axis': col,
                'feature': feature
            }

            feature_vector = []
            for level in range(1, max_level + 1):
                nodes = wp.get_level(level, ordering)

                if feature == 'energy':
                    e = [energy(node.data) for node in nodes]
                    feature_vector.extend(e)

                elif feature == 'energy_ratio':
                    e = [energy(node.data) for node in nodes]
                    total_energy = np.sum(e)
                    energy_ratios = [energy(node.data) / total_energy for node in nodes]
                    feature_vector.extend(energy_ratios)

                elif feature == 'kurtosis':
                    kurts = [ft.kurtosis(node.data) for node in nodes]
                    feature_vector.extend(kurts)

                elif feature == 'negentropy':
                    negentropies = [negentropy(node.data) for node in nodes]
                    feature_vector.extend(negentropies)

                wpd_header.extend([f'L{level}_{i}' for i in range(len(nodes))])

            row.update(dict(zip(wpd_header, feature_vector)))
            result.append(row)
    return result


def list_files(dataset: ZipFile) -> List[str]:
    """List csv and tsv files in dataset within ZIP archive

    :param dataset: dataset in zip archive

    :return: list of filenames
    """
    filenames = [
        f.filename
        for f in dataset.infolist()
        if f.filename.endswith(('.csv', '.tsv'))
    ]
    filenames.sort()
    return filenames


def fs_list_files(root_path: str) -> List[str]:
    """List csv and tsv files in dataset in the directory

    :param root_path: base path of dataset directory

    :return: list of filenames
    """
    filenames = [
        os.path.join(root, filename)
        for root, dirs, files in os.walk(root_path)
        for filename in files
        if filename.endswith(('.csv', '.tsv'))

    ]
    filenames.sort()
    return filenames


def load_files_split(
            dataset: ZipFile,
            func: Callable,
            parts: int = 1,
            cores: int = 4
        ) -> pd.DataFrame:
    """Load files from the dataset in ZIP archive and process them with callback function

    :param dataset: dataset in zip archive
    :param func: callback for file processing that takes dataset, filename, and number of parts
        to split file into as parameters
    :param parts: number of partitions to split each file into
    :param cores: number of cores to use in workload parallelization

    :return: data frame with extracted features and associated annotations for row
    """
    pool = ThreadPool(processes=cores)
    filenames = list_files(dataset)

    return pd.concat([
        pool.apply_async(func, (dataset, name, parts)).get()
        for name in tqdm(filenames)
    ])


def split_dataframe(dataframe: pd.DataFrame, parts: int = None) -> List[pd.DataFrame]:
    """Split to data frames to non overlapping parts

    :param dataframe: data frame to be split
    :param parts: number of partitions to split data frame into

    :return: list of data frame parts
    """
    if parts is None:
        return [dataframe]

    step = len(dataframe) // parts
    return [
        dataframe.iloc[i:i+step].reset_index(drop=True)
        for i in range(0, len(dataframe), step)
        if len(dataframe.iloc[i:i + step]) == step
    ]


def detrending_filter(
            dataframes: List[pd.DataFrame],
            columns: List[str]
        ) -> List[pd.DataFrame]:
    """Subtract average from each value in columns of multiple data frames

    :param dataframes: list of data frames
    :param columns: attributes from which mean is removed

    :return: modified data frames with mean removed
    """
    for df in dataframes:
        df[columns] = df[columns].apply(lambda x: x - x.mean())
    return dataframes


def load_features(
            filename: str,
            axis: List[str],
            label_columns: List[str]
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load features from csv file and aggregate values from chosen directions of movement

    :param filename: File path where extracted features are found
    :param axis: Elements to combine one feature from
    :param label_columns: Columns that are not features and are not processed just copied

    :return: data frame with columns of unique feature names
    """
    features = pd.read_csv(filename)

    columns = features.columns.str.startswith(tuple(axis))
    X = features[features.columns[columns]]
    Y = features[label_columns]
    feature_names = X.columns.str.extract(r'([a-z]+)_([a-z\_\-]+)')[1].unique()

    df = pd.DataFrame()
    for name in feature_names:
        vector_dims = [f'{dim}_{name}' for dim in axis]
        df[name] = X[vector_dims].apply(np.linalg.norm, axis=1)

    df[label_columns] = Y
    return df


def mms_peak_finder(x: np.array, win_len: int = 3) -> np.array:
    """Robust non-parametric peak identification MMS algorithm
    according to description in paper: "Non-Parametric Local Maxima and Minima
    Finder with Filtering Techniques for Bioprocess"
    (https://doi.org/10.4236/jsip.2016.74018)

    :param x: time series
    :param win_len: window length of points compared in peak finding

    :return: array of indexes where peaks are in the time series
    """
    a = sliding_window_view(x, window_shape=win_len)

    mms_max = (
        (np.max(a, axis=1) - np.min(a, axis=1)) /
        (np.sum(a, axis=1) - np.min(a, axis=1) * win_len)
    )
    mms_mid = (
        (a[:,win_len//2] - np.min(a, axis=1)) /
        (np.sum(a, axis=1) - np.min(a, axis=1) * win_len)
    )
    peaks_in_windows, *other = np.where(mms_max == mms_mid)
    return peaks_in_windows + 1


def have_intersection(interval1: Tuple[float, float], interval2: Tuple[float, float]) -> bool:
    """Check overlap of two numeric intervals

    :param interval1: numbers for bounds of the first interval
    :param interval2: numbers for bounds of the second interval

    :return: overlap of intervals is present
    """
    new_min = max(interval1[0], interval2[0])
    new_max = min(interval1[1], interval2[1])
    return new_min <= new_max


def downsample(x: np.array, k: int, fs_reduced: int, fs: int) -> np.array:
    """Downsample the time series by a factor

    :param x: time series
    :param k: factor for downsampling (when it is None, the factor is calculated as a ratio of sampling rates)
    :param fs_reduced: desired sampling frequency
    :param fs: original sampling frequency of the time series
    :return: downsampled time series
    """
    if k is None:
        k = fs // fs_reduced
    return decimate(x, k, ftype='iir')


def harmonic_series_detection(
            f: np.array,
            pxx: np.array,
            fs: int,
            fft_window: int
        ) -> List[List[Tuple[int, float]]]:
    """Find all series of harmonic frequencies in the frequency spectrum according to paper:
    "Identification of harmonics and sidebands in a finite set of spectral components"
    (https://hal.science/hal-00845120v2/document)

    :param f: array of frequency bins
    :param pxx: array of amplitudes at frequency bins
    :param fs: sampling frequency in Hz
    :param fft_window: length of FFT window for calculation of frequency bin resolution
    :return: list of harmonic frequency series with components described by their frequency and amplitude
    """
    peaks = mms_peak_finder(pxx)
    f_central = f[peaks]
    delta_f = fs / fft_window
    amplitudes = pxx[peaks]

    largest_frequency = int(fs / 2 + delta_f)
    components = np.vstack((f_central, amplitudes)).T
    k = 8     # Limit harmonics not detected in series (skips)

    result = []
    # Each component can be fundamental frequency
    for v, a in components:
        series = [(v, a)]

        # Harmonic components from given fundamental frequency
        order_distance = 0
        delta_v = delta_f
        for r, vi in enumerate(range(2*int(v), largest_frequency, int(v)), start=2):
            # Search interval for harmonic component candidates
            search_interval = (vi - ((r * delta_v) / 2), vi + ((r * delta_v) / 2))

            # If tolerances are in search interval find minimal distance
            # between true harmonic and candidate
            candidates = []
            for vj, aj in components:
                tolerance_interval = (vj - delta_v / 2, vj + delta_v / 2)
                if have_intersection(search_interval, tolerance_interval):
                    candidates.append((vj, aj, abs(vj - vi)))

            if len(candidates) > 0:
                vh, ah, dh = min(candidates, key=lambda x: x[2])
                series.append((vh, ah))
                xh, yh = (vh - delta_v), (vh + delta_v)
                # Update parameters to prevent of search interval growth
                vi = (xh + yh) / (2*r)
                delta_v = abs(xh - yh) / r

            if order_distance == k:
                break

        if len(series) > 1:
            result.append(series)

    return result
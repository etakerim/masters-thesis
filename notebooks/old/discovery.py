from zipfile import ZipFile
import numpy as np
import pandas as pd
from itertools import pairwise
from typing import List, Tuple, Callable

from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch, windows, find_peaks
from scipy.interpolate import interp1d
from scipy.fft import rfft

from scipy.signal import butter, iirfilter, freqz, lfilter, decimate
import pywt
from tsfel import feature_extraction as ft
import matplotlib.pylab as plt



def plot_label_occurences(y):
    observations = []
    columns = list(y.astype('category').cat.categories)
    empty = dict(zip(columns, len(columns) * [0]))

    for row in y.astype('category'):
        sample = empty.copy()
        sample[row] = 1
        observations.append(sample)

    class_occurences = pd.DataFrame.from_records(observations).cumsum()
    ax = class_occurences.plot(grid=True, figsize=(10, 5), xlabel='Observations', ylabel='Label occurences')
    return ax, class_occurences


def features_list():
    config = tsfel.get_features_by_domain()
    for domain in config.values():
        for feature, options in domain.items():
            if options['n_features'] != 1:
                options['use'] = 'no'
    return config


def features_wavelet_domain(zip_file: ZipFile, filename: str) -> pd.DataFrame:
    print(f'Processing: {filename}')

    max_level = 6
    wavelet = 'dmey'
    ordering = 'freq'

    columns = mafaulda.COLUMNS
    ts = mafaulda.csv_import(zip_file, filename)
    fault, severity, seq = mafaulda.parse_filename(filename)

    rpm = ts['rpm'].mean()
    result = []

    for col in columns:
        wp = pywt.WaveletPacket(data=ts[col], wavelet=wavelet, mode='symmetric')

        for feature in ('energy', 'energy_ratio', 'kurtosis', 'negentropy'):
            wpd_header = []
            row = {
                'fault': fault,
                'severity': severity,
                'seq': seq,
                'rpm': rpm,
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
                    kurts = [kurtosis(node.data) for node in nodes]
                    feature_vector.extend(kurts)

                elif feature == 'negentropy':
                    negentropies = [negentropy(node.data) for node in nodes]
                    feature_vector.extend(negentropies)

                wpd_header.extend([f'L{level}_{i}' for i in range(len(nodes))])

            row.update(dict(zip(wpd_header, feature_vector)))
            result.append(row)

    return pd.DataFrame(result).reset_index(drop=True)



def dc_blocker(x: np.array, cutoff: float, order, fs, plot=False):
    b, a = iirfilter(1, cutoff, btype='highpass', fs=fs)
    if plot:
        plot_filter_response(b, a)
    y = lfilter(b, a, x)
    return y


def downsample(x: np.array, k, fs_reduced, fs):
    if k is None:
        k = fs // fs_reduced
    return decimate(x, k, ftype='iir')


def lowpass_filter(x: np.array, cutoff: float, order, fs, plot=False):
    b, a = butter(2, cutoff, btype='lowpass', fs=fs)
    if plot:
        plot_filter_response(b, a)
    y = lfilter(b, a, x)
    return y


def plot_filter_response(b, a, title=''):
    w, h = freqz(b, a, fs=50000)
    fig, ax = plt.subplots()
    ax.plot(w, 20 * np.log10(abs(h)), 'b')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [dB]')
    ax.set_title(title)


###############################################################

def mms_peak_finder(x: np.array, win_len=3) -> np.array:
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


def have_intersection(interval1, interval2):
    new_min = max(interval1[0], interval2[0])
    new_max = min(interval1[1], interval2[1])
    return new_min <= new_max


def harmonic_series_detection(f: np.array, Pxx: np.array, fs: int, fft_window: int) -> np.array:
    peaks = mms_peak_finder(Pxx)
    f_central = f[peaks]
    delta_f = fs / fft_window
    amplitudes = Pxx[peaks]

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


#################################################################

def find_harmonics(f: np.array, Pxx: np.array) -> (np.array, np.array):
    threshold = Pxx.mean() +  2*np.std(Pxx)
    peaks, _ = find_peaks(Pxx)
    f_harmonics = f[peaks]
    y_harmonics = Pxx[peaks]

    cond = y_harmonics >= threshold
    loc_harmonics = peaks[cond]
    f_harmonics = f_harmonics[cond]
    return loc_harmonics, f_harmonics



DB_REF = 0.000001                                # 1 dB = 1 um/s^2

def resolution_calc(fs, window):
    print('Window size:', window)
    print('Heinsenberg box')
    print('\tTime step:', window / fs * 1000, 'ms')
    print('\tFrequency step:', fs / window, 'Hz')

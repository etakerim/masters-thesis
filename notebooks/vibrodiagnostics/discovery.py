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
from vibrodiagnostics import mafaulda



def split_dataframe(dataframe: pd.DataFrame, parts: int = None) -> List[pd.DataFrame]:
    if parts is None:
        return [dataframe]

    step = len(dataframe) // parts
    return [
        dataframe.iloc[i:i+step].reset_index(drop=True)
        for i in range(0, len(dataframe), step)
        if len(dataframe.iloc[i:i + step]) == step
    ]

def detrending_filter(dataframe: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for df in dataframe:
        df[columns] = df[columns].apply(lambda x: x - x.mean())
    return dataframe


def butter_bandpass_filter(data, cutoff=10000, fs=mafaulda.FS_HZ, order=5):
    b, a = butter(order, cutoff, fs=fs, btype='lowpass')
    y = lfilter(b, a, data.to_numpy())
    return pd.Series(data=y, index=data.index)


def lowpass_filter_extract(dataframe: pd.DataFrame, columns) -> pd.DataFrame:
    for df in dataframe:
        df[columns] = df[columns].apply(butter_bandpass_filter)
    return dataframe


def time_features_calc(df: pd.DataFrame, col: str) -> List[Tuple[str, pd.DataFrame]]:
    x = df[col]
    features = [
        ('std', [ft.calc_std(x)]),
        ('skewness', [ft.skewness(x)]),
        ('kurtosis', [ft.kurtosis(x)]),
        ('rms', [ft.rms(x)]),
        ('pp', [ft.pk_pk_distance(x)]),
        ('crest', [np.max(np.absolute(x)) / ft.rms(x)]),
        ('margin', [np.max(np.absolute(x)) / (np.mean(np.sqrt(np.absolute(x))) ** 2)]),
        ('impulse', [np.max(np.absolute(x)) / np.mean(np.absolute(x))]),
        ('shape', [ft.rms(x) / np.mean(np.absolute(x))]),
        ('max', [ft.calc_max(x)])
    ]
    return [(f'{col}_{f[0]}', f[1]) for f in features]

def frequency_features_calc(df: pd.DataFrame, col: str, window: int) -> List[Tuple[str, pd.DataFrame]]:
    f, pxx = spectral_transform(df, col, window)
    
    fluxes = temporal_variation(df, col, window)
    envelope_spectrum = envelope_signal(f, pxx)
    # loc_harmonics, _ = find_harmonics(f, pxx)

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
    return [(f'{col}_{f[0]}_{window}', f[1]) for f in features]


############### BACKUP ########################xxx
def tsfel_features_import(zip_file: ZipFile, filename: str, parts=None) -> pd.DataFrame:
    ts = mafaulda.csv_import(zip_file, filename)
    cfg_file = tsfel.get_features_by_domain()
    dataframe = detrending_filter(split_dataframe(ts, parts), columns)
        
    result = []
    for i, df in enumerate(dataframe):
        df_result = tsfel.time_series_features_extractor(cfg_file, df[mafaulda.COLUMNS], fs=mafaulda.FS_HZ)
        result.append(df_result)

    return pd.concat(result).reset_index(drop=True)

def me_tsfel_features_import(filename: str, loader: Callable, parts: int=None) -> pd.DataFrame:
    print(f'Processing: {filename}')
    name, ts, fs_hz, columns = loader(filename)
    cfg_file = tsfel.get_features_by_domain()
    df = detrending_filter(split_dataframe(ts, parts), ['x', 'y', 'z'])
        
    result = []
    for i, df in enumerate(dataframe):
        df_result = tsfel.time_series_features_extractor(cfg_file, df, fs=fs_hz)
        result.append(df_result)

    return pd.concat(result).reset_index(drop=True)


#################################################xxx

def plot_label_occurences(y):
    observations = []
    columns = list(y.astype('category').cat.categories)
    empty = dict(zip(columns, len(columns) * [0]))

    for row in y.astype('category'):
        sample = empty.copy()
        sample[row] = 1
        observations.append(sample)

    class_occurences = pd.DataFrame.from_records(observations).cumsum()
    class_occurences.plot(grid=True, figsize=(10, 3), xlabel='Observations', ylabel='Label occurences')


def features_list():
    config = tsfel.get_features_by_domain()
    for domain in config.values():
        for feature, options in domain.items():
            if options['n_features'] != 1:
                options['use'] = 'no'
    return config


def tsfel_features_generate(dataset: ZipFile, filename: str, parts=None) -> pd.DataFrame:
    print(f'Processing: {filename}')

    conf_extraction = features_list()
    columns = mafaulda.COLUMNS

    ts = mafaulda.csv_import(dataset, filename)
    fault, severity, seq = mafaulda.parse_filename(filename)
    dataframe = [ts] if parts is None else discovery.split_dataframe(ts, parts)

    results = []
    for i, df in enumerate(dataframe):
        fv = pd.DataFrame({
            'fault': [fault],
            'severity': [severity],
            'seq': [f'{seq}.part.{i}'],
            'rpm': [df['rpm'].mean()]
        })

        for col in columns:
            features = tsfel.time_series_features_extractor(
                conf_extraction, df[col], fs=mafaulda.FS_HZ
            )
            features.columns = [
                col + '_' + c.strip('0_').replace(' ', '_').lower()
                for c in features.columns
            ]
            fv = fv.assign(**features)
        results.append(fv)

    return pd.concat(results).reset_index(drop=True)


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


###############################################################
# https://dsp.stackexchange.com/questions/19084/applying-filter-in-scipy-signal-use-lfilter-or-filtfilt

def dc_blocker(x: np.array, cutoff: float, order=1, fs=mafaulda.FS_HZ, plot=False):
    b, a = iirfilter(order, cutoff, btype='highpass', fs=fs)
    if plot:
        plot_filter_response(b, a)
    y = lfilter(b, a, x)
    return y


def downsample(x: np.array, k=None, fs_reduced=mafaulda.FS_HZ, fs=mafaulda.FS_HZ):
    if k is None:
        k = fs // fs_reduced
    return decimate(x, k, ftype='iir')


def lowpass_filter(x: np.array, cutoff: float, order=2, fs=mafaulda.FS_HZ, plot=False):
    b, a = butter(order, cutoff, btype='lowpass', fs=fs)
    if plot:
        plot_filter_response(b, a)
    y = lfilter(b, a, x)
    return y


def plot_filter_response(b, a, title=''):
    w, h = freqz(b, a, fs=mafaulda.FS_HZ)
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


def envelope_signal(f: np.array, Pxx: np.array) -> np.array:
    peaks, _ = find_peaks(Pxx)
    # peaks = mms_peak_finder(Pxx)
    try:
        envelope = interp1d(f[peaks], Pxx[peaks], kind='quadratic', fill_value='extrapolate')
    except ValueError:
        return []
    y_env = envelope(f)
    y_env[y_env < 0] = 0
    return y_env


def spectral_transform(dataset: pd.DataFrame, axis: str, window: int) -> (np.array, np.array):
    OVERLAP = 0.5
    STEP = int(window * OVERLAP)
    v = dataset[axis].to_numpy()
    f, Pxx = welch(
        v, fs=mafaulda.FS_HZ, window='hann',
        nperseg=window, noverlap=STEP,
        scaling='spectrum', average='mean', detrend='constant',
        return_onesided=True
    )
    return f, Pxx

###################################################################x

def energy(Pxx: np.array) -> float:
    return np.sum(Pxx**2)


def negentropy(x: np.array) -> float:
    if len(x) == 0:
        return np.nan
    return -entropy((x ** 2) / np.mean(x ** 2))


def signal_to_noise(x: np.array) -> float:
    # https://dsp.stackexchange.com/questions/76291/how-to-extract-noise-from-a-signal-in-order-to-get-both-noise-power-and-signal-p
    # https://www.geeksforgeeks.org/signal-to-noise-ratio-formula/
    # https://saturncloud.io/blog/calculating-signaltonoise-ratio-in-python-with-scipy-v11/
    m = np.mean(x)
    sd = np.std(x)
    return np.where(sd == 0, 0, m / sd)


def spectral_roll_off_frequency(f: np.array, Pxx: np.array, percentage: float) -> float:
    # Roll-off: Cumulative sum of energy in spectral bins and find index in f array
    # 95% of total energy below this frequency
    return f[np.argmax(np.cumsum(Pxx**2) >= percentage * energy(Pxx))]


def temporal_variation(dataset: pd.DataFrame, axis: str, window: int) -> list:
    # Temporal variation of succesive spectra (stationarity)
    OVERLAP = 0.5
    STEP = int(window * OVERLAP)
    v = dataset[axis].to_numpy()
    spectra = [
        np.absolute(rfft(v[i:i+window] * windows.hann(window)))
        for i in range(0, len(v) - window, STEP)
    ]
    # f = [i * (mafaulda.FS_HZ / window) for i in range(window // 2 + 1)]
    fluxes = [
        1 - np.corrcoef(psd1, psd2) for psd1, psd2 in pairwise(spectra)
    ]
    return fluxes


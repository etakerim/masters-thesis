from zipfile import ZipFile
import numpy as np
import pandas as pd
from itertools import pairwise

from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch, windows, find_peaks
from scipy.interpolate import interp1d
from scipy.fft import rfft

from scipy.signal import butter, iirfilter, freqz, lfilter, decimate
import pywt

import matplotlib.pylab as plt
from feature import mafaulda


def features_time_domain(zip_file: ZipFile, filename: str) -> pd.DataFrame:
    print(f'Processing: {filename}')

    columns = mafaulda.COLUMNS
    ts = mafaulda.csv_import(zip_file, filename)
    fault, severity, seq = mafaulda.parse_filename(filename)

    rpm = ts['rpm'].mean()

    feature_vector = [
        ts[columns].mean().rename('mean'),
        ts[columns].std().rename('std'),
        ts[columns].apply(lambda x: skew(x)).rename('skew'),
        ts[columns].apply(lambda x: kurtosis(x)).rename('kurt'),
        ts[columns].apply(mafaulda.rms).rename('rms'),
        ts[columns].apply(lambda x: np.max(x) - np.min(x)).rename('pp'),
        ts[columns].apply(lambda x: np.max(np.absolute(x)) / mafaulda.rms(x)).rename('crest'),
        ts[columns].apply(lambda x: np.max(np.absolute(x)) / np.mean(np.sqrt(np.absolute(x))) ** 2).rename('margin'),
        ts[columns].apply(lambda x: np.max(np.absolute(x)) / np.mean(np.absolute(x))).rename('impulse'),
        ts[columns].apply(lambda x: mafaulda.rms(x) / np.mean(np.absolute(x))).rename('shape'),
        ts[columns].max().rename('max'),
    ]
    return (
        pd.concat(feature_vector, axis=1)
        .assign(fault=fault, severity=severity, seq=seq, rpm=rpm)
        .reset_index()
        .rename(columns={'index': 'axis'})
    )


def features_frequency_domain(zip_file: ZipFile, filename: str) -> pd.DataFrame:
    # Calculate FFT with Welch method in 5 different Hann window sizes
    print(f'Processing: {filename}')

    OVERLAP = 0.5
    WINDOW_SIZES = (2**8, 2**10, 2**12, 2**14, 2**16)

    columns = mafaulda.COLUMNS
    ts = mafaulda.csv_import(zip_file, filename)
    fault, severity, seq = mafaulda.parse_filename(filename)

    rpm = ts['rpm'].mean()

    result = []

    for window in WINDOW_SIZES:
        for col in columns:
            f, Pxx = spectral_transform(ts, col, window)

            fluxes = temporal_variation(ts, col, window)
            envelope_spectrum = envelope_signal(f, Pxx)
            loc_harmonics, _ = find_harmonics(f, Pxx)

            result.append({
                'fft_window_length': window,
                'fault': fault,
                'severity': severity,
                'seq': seq,
                'rpm': rpm,
                'axis': col,
                'centroid': np.average(f, weights=Pxx),
                'std': np.std(Pxx),
                'skew': skew(Pxx),
                'kurt': kurtosis(Pxx),
                'roll-off': spectral_roll_off_frequency(f, Pxx, 0.85),
                'flux_mean': np.mean(fluxes),
                'flux_std': np.std(fluxes),
                'hdev': hdev(envelope_spectrum, loc_harmonics, Pxx),
                'noisiness': signal_to_noise(Pxx),
                'inharmonicity': inharmonicity(loc_harmonics, f, Pxx),
                'energy': energy(Pxx),
                'entropy': entropy(Pxx / np.sum(Pxx)),
                'negentropy': negentropy(envelope_spectrum)
            })

    return pd.DataFrame(result)


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

    return pd.DataFrame(result)


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


def hdev(envelope_spectrum: np.array, loc_harmonics: np.array, Pxx: np.array) -> float:
    if len(envelope_spectrum) == 0 or len(envelope_spectrum) == 0:
        return np.nan
    return np.mean(envelope_spectrum[loc_harmonics] - Pxx[loc_harmonics])


def signal_to_noise(x: np.array) -> float:
    # https://dsp.stackexchange.com/questions/76291/how-to-extract-noise-from-a-signal-in-order-to-get-both-noise-power-and-signal-p
    # https://www.geeksforgeeks.org/signal-to-noise-ratio-formula/
    # https://saturncloud.io/blog/calculating-signaltonoise-ratio-in-python-with-scipy-v11/
    m = np.mean(x)
    sd = np.std(x)
    return np.where(sd == 0, 0, m / sd)


def harmonic_product_spectrum(f: np.array, Pxx: np.array, max_harmonic=5) -> float:
    # Returns fundamental frequency = pitch
    # http://musicweb.ucsd.edu/~trsmyth/analysis/Harmonic_Product_Spectrum.html
    hps_spectrum = np.copy(Pxx)
    # Downsampling in factor of 2 .. h
    for harmonic in range(2, max_harmonic + 1):
        hps_spectrum[:len(Pxx[::harmonic])] *= Pxx[::harmonic]

    fundamental = np.argmax(hps_spectrum)
    return fundamental, f[fundamental]


def inharmonicity(loc_harmonics: np.array, f: np.array, Pxx: np.array) -> float:
    f0_pos, f0 = harmonic_product_spectrum(f, Pxx)
    if f0 == 0 or len(loc_harmonics) == 0 or np.sum(Pxx[loc_harmonics]) == 0:
        return np.nan

    harmonic_series = np.arange(f0, f0 * (len(loc_harmonics) + 1), f0)

    return (2 / f0) * np.average(np.absolute(f[loc_harmonics] - harmonic_series), weights=Pxx[loc_harmonics])


def spectral_roll_off_frequency(f: np.array, Pxx: np.array, percentage: float) -> float:
    # Roll-off: Cumulative sum of energy in spectral bins and find index in f array
    # 85% of total energy below this frequency
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

###############################################################################
def plot_spectral_envelope(dataset, file, axis):
    ts = mafaulda.csv_import(dataset, file)
    f, Pxx = spectral_transform(ts, axis, 1024)
    y_env = envelope_signal(f, Pxx)
    # print(find_harmonics(f, Pxx))

    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.plot(f, Pxx)
    #ax.scatter(f[peaks], Pxx[peaks], color='red')
    ax.plot(f, y_env)


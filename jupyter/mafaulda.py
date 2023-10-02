import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import skew, kurtosis, kstest
from scipy.signal import welch
from scipy.fft import rfft
from scipy.signal import butter, filtfilt, find_peaks
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from multiprocessing.pool import ThreadPool

FS_HZ = 50000


def import_files(zip_file, file_list, func, cores=4):
    pool = ThreadPool(processes=cores)
    
    return pd.concat([
        pool.apply_async(func, (zip_file, name, )).get()
        for name in tqdm(file_list)
    ])

def normality_tests(ts, columns=None):
    columns = columns or ['ax', 'ay', 'az', 'bx', 'by', 'bz']
    figure, axes = plt.subplots(2, 3, figsize=(10, 5))

    for i, col in enumerate(columns):
        print('Normality test p-value: ', kstest(ts[col], 'norm').pvalue, '(<0.05 is not normal)')
        sm.qqplot(ts[col], line='45', ax = axes[i // 3, i % 3])

    plt.tight_layout()
    plt.show()


def resolution_calc(fs, window):
    print('Window size:', window)
    print('Heinsenberg box')
    print('\tTime step:', window / fs * 1000, 'ms')
    print('\tFrequency step:', fs / window, 'Hz')


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def get_mafaulda_files(zip_file):
    filenames = [
        text_file.filename
        for text_file in zip_file.infolist()
        if text_file.filename.endswith('.csv')
    ]
    filenames.sort()
    return filenames


def preprocess(ts):
    return (
        ts
        .assign(t = lambda x: x.index * (1 / FS_HZ))
        .assign(mag_a = lambda x: np.hypot(x.ax, x.ay, x.ay))
        .assign(mag_b = lambda x: np.hypot(x.bx, x.by, x.by))
        .assign(rev = lambda x: (x.tachometer - x.shift(-1).tachometer) >= 3)
        .assign(rpm = lambda x: 60 / (x[x.rev == True].t - x[x.rev == True].shift(1).t))
        .assign(rpm = lambda x: x.rpm.ffill().rolling(
            (x[x.rev == True].index.values - np.roll(x[x.rev == True].index.values, 1)).max()
        ).median())  # Smooth out outliers by robust filter
        .dropna()
        .reset_index(drop=True)
        .assign(t = lambda x: x.index * (1 / FS_HZ))
        .set_index('t')
    )



def csv_import(zip_file, filename):
    ts = pd.read_csv(
        zip_file.open(filename),
        names=['tachometer', 'ax', 'ay', 'az', 'bx', 'by', 'bz', 'mic']
    )
    ts = preprocess(ts)
    return ts.assign(key=filename)


def fft_csv_import(zip_file, filename, window=4096, overlap=0.5, fs=50000, is_welch=False):
    STEP = window * overlap
    col = 'ax'
    info = filename.split('/')
    load = int(info[1].strip(' g'))

    frame = csv_import(zip_file, filename)
    v = frame[col].to_numpy()

    if is_welch is False:
        spectra = [
            np.abs(rfft(v[i:i+window] * np.hamming(window)))
            for i in range(0, len(v) - window, int(STEP))
        ]
        freqs = [i * (fs / window) for i in range(window // 2 + 1)]

    else:
        freqs, spectra = welch(v, fs, 'hann', nperseg=window, scaling='spectrum', average='mean')
        spectra = [spectra]


    return (
        pd.DataFrame(data=spectra, columns=freqs.astype(int))
        .assign(load=load, no=info[2])
        .set_index(['load', 'no'])
    )


def fft_csv_import_by_axis(zip_file, filename, axis='ax', window=4096, overlap=0.5, fs=50000, is_welch=False):
    STEP = window * overlap

    frame = csv_import(zip_file, filename)
    v = frame[axis].to_numpy()

    if is_welch is False:
        spectra = [
            np.abs(rfft(v[i:i+window] * np.hamming(window)))
            for i in range(0, len(v) - window, int(STEP))
        ]
        freqs = [i * (fs / window) for i in range(window // 2 + 1)]

    else:
        freqs, spectra = welch(v, fs, 'hann', nperseg=window, scaling='spectrum', average='mean')
        spectra = [spectra]


    return (
        pd.DataFrame(data=spectra, columns=freqs.astype(int))
        .assign(name=filename, rpm=frame['rpm'].median())
        .set_index(['name'])
    )


def time_domain_features(ts, col):
    return pd.concat([
            ts.groupby(by='key')[col].mean().rename('mean'),
            ts.groupby(by='key')[col].std().rename('std'),
            ts.groupby(by='key')[col].apply(lambda x: skew(x)).rename('skew'),
            ts.groupby(by='key')[col].apply(lambda x: kurtosis(x)).rename('kurtosis'),
            ts.groupby(by='key')[col].apply(rms).rename('rms')
        ],
        axis=1
    )


def extract_peaks(psd, max_freq=1000):
    MAX_FREQ = 1000
    frames = []

    for index, bins in psd.iterrows():
        peaks, properties = find_peaks(bins[:max_freq], prominence=0.02)
        row = {
            'load': index[0],
            'no': index[1],
            'f': bins.index[peaks],
            'y': bins[bins.index[peaks]]
        }
    
        frame = pd.DataFrame(data=row, columns=['load', 'no', 'f', 'y'])
        frames.append(frame)
        
    harmonics = (
        pd.concat(frames)
          .sort_values(by=['load', 'y', 'f'], ascending=[True, False, True])
    )
    
    f0 = harmonics.groupby('load').nth(0)
    f1 = harmonics.groupby('load').nth(1)
    peak_features = f0.join(f1, lsuffix='_f0', rsuffix='_f1').reset_index()
    return peak_features


def csv_import_td_features(zip_file, filename, col='ax'):
    frame = csv_import(zip_file, filename)
    info = filename.split('/')
    frame = frame.assign(load=int(info[1].strip(' g')), no=info[2])

    return pd.concat([
            frame.groupby(by=['load', 'no'])[col].mean().rename('mean'),
            frame.groupby(by=['load', 'no'])[col].std().rename('std'),
            frame.groupby(by=['load', 'no'])[col].apply(lambda x: skew(x)).rename('skew'),
            frame.groupby(by=['load', 'no'])[col].apply(lambda x: kurtosis(x)).rename('kurtosis'),
            frame.groupby(by=['load', 'no'])[col].apply(rms).rename('rms'),
            frame.groupby(by=['load', 'no'])[col].apply(lambda x: max(abs(x.max()), abs(x.min()))).rename('amplitude')
        ],
        axis=1
    ).reset_index()


def axis_spectrograms(df):
    fig, ax = plt.subplots(4, 1, figsize=(20, 10))

    RESOLUTION = 8
    WINDOW = FS_HZ // RESOLUTION
        
    resolution_calc(FS_HZ, WINDOW)

    for i, col in enumerate(['ax', 'ay', 'az']):
        pxx, freq, t, cax = ax[i].specgram(
            df[col],
            Fs=FS_HZ, 
            mode='magnitude',
            window=np.hamming(WINDOW), 
            NFFT=WINDOW, 
            noverlap=WINDOW//2
        )
    
    pxx, freq, t, cax = ax[3].specgram(
        df.mag_b,
        Fs=FS_HZ, 
        mode='magnitude',
        window=np.hamming(WINDOW), 
        NFFT=WINDOW, 
        noverlap=WINDOW//2
    )
    
    for i in range(3):
        ax[i].set_ylabel('Frequency [Hz]')
    ax[2].set_xlabel('Time [s]')
    
    g = plt.colorbar(cax, ax=ax)


def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def rms_orbitals(ts, n=100):
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    
    ax_rms = ts['ax'].rolling(n).apply(rms)
    ay_rms = ts['ay'].rolling(n).apply(rms)
    az_rms = ts['az'].rolling(n).apply(rms)

    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].scatter(ax_rms, ay_rms, s=1)
    
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('z')
    ax[1].scatter(ax_rms, az_rms, s=1)
    
    ax[2].set_xlabel('y')
    ax[2].set_ylabel('z')
    ax[2].scatter(ay_rms, az_rms, s=1)

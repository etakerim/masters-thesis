import numpy as np
import pandas as pd
import seaborn as sb

from scipy.stats import skew, kurtosis, kstest
from scipy.fft import rfft
from scipy.signal import butter, filtfilt, find_peaks

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from zipfile import ZipFile

from multiprocessing.pool import ThreadPool

FS_HZ = 50000
DB_REF = 0.000001                                # 1 dB = 1 um/s^2
COLUMNS = ['ax', 'ay', 'az', 'bx', 'by', 'bz']   # 'mic'

################################# IMPORT ########################################

def parse_filename(filename: str) -> (str, str, str):
    path = filename.split('/')
    if path[0].strip() in ('overhang', 'underhang'):
        fault = f'{path[0]}-{path[1]}'
        severity = path[2]
        seq = path[3]
    elif path[0].strip() == 'normal':
        fault, severity, seq = path[0], '0', path[1]
    else:
        fault, severity, seq = path

    return fault, severity, seq


def import_files_split(zip_file, file_list, func, parts, cores=4):
    pool = ThreadPool(processes=cores)
    
    return pd.concat([
        pool.apply_async(func, (zip_file, name, parts )).get()
        for name in tqdm(file_list)
    ])


def import_files(zip_file, file_list, func, cores=4):
    pool = ThreadPool(processes=cores)
    
    return pd.concat([
        pool.apply_async(func, (zip_file, name, )).get()
        for name in tqdm(file_list)
    ])


def get_mafaulda_files(zip_file):
    filenames = [
        text_file.filename
        for text_file in zip_file.infolist()
        if text_file.filename.endswith('.csv')
    ]
    filenames.sort()
    return filenames


def rpm_calc(tachometer: pd.Series) -> float:
    t = tachometer.index.to_numpy()
    y = tachometer.to_numpy()
    peaks, _ = find_peaks(y, prominence=3, width=50)
    interval = np.diff(t[peaks]).mean()
    return 60 / interval


def preprocess(ts: pd.DataFrame):
    return (
        ts
        .assign(t = lambda x: x.index * (1 / FS_HZ))
        # .assign(mag_a = lambda x: np.hypot(x.ax, x.ay, x.ay))
        # .assign(mag_b = lambda x: np.hypot(x.bx, x.by, x.by))
        .reset_index()
        .assign(t = lambda x: x.index * (1 / FS_HZ))
        .set_index('t')
        .assign(rpm = lambda x: rpm_calc(x.tachometer))
    )


def csv_import(zip_file, filename):
    ts = pd.read_csv(
        zip_file.open(filename),
        names=['tachometer', 'ax', 'ay', 'az', 'bx', 'by', 'bz', 'mic']
    )
    ts = preprocess(ts)
    return ts.assign(key=filename)

################################# DATASET INDEX ########################################

def extract_metadata(zip_file: ZipFile, filename: str) -> pd.DataFrame:
    ts = csv_import(zip_file, filename)
    fault, severity, seq = parse_filename(filename)
    result = [{
        'filename': filename,
        'fault': fault,
        'severity': severity,
        'seq': seq,
        'length': len(ts.index),
        'duration': ts.index.max(), 
        'rpm': ts['rpm'].mean(),
        'fs': FS_HZ,
    }]
    return pd.DataFrame(result)

def dataset_index(path: str) -> pd.DataFrame:
    dataset = ZipFile(path)
    files = get_mafaulda_files(dataset)
    return import_files(dataset, files, extract_metadata)


def resolution_calc(fs, window):
    print('Window size:', window)
    print('Heinsenberg box')
    print('\tTime step:', window / fs * 1000, 'ms')
    print('\tFrequency step:', fs / window, 'Hz')


def rms(x):
    return np.sqrt(np.mean(x ** 2))

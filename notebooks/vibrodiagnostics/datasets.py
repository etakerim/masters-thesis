import os
from typing import List, Tuple
import pandas as pd

from tqdm.notebook import tqdm
from multiprocessing.pool import ThreadPool


def steval_files(path: str):
    dataset = []
    for root, subdirs, files in os.walk(path):
        for filename in files:
            if filename.lower().endswith('.tsv'):
                dataset.append(os.path.join(root, filename))
    return dataset


def steval_measurement(filename: str, fs: int=26866) -> Tuple[str, pd.DataFrame]:
    g = 9.81
    columns = ['x', 'y', 'z']
    ts = pd.read_csv(
        filename,
        delimiter='\t',
        index_col=False,
        header=0,
        names=['t'] + columns
    )

    for dim in columns:
        ts[dim] -= ts[dim].mean()
        ts[dim] = g * (ts[dim] / 1000)

    ts['t'] = ts.index * (1 / fs)
    ts.set_index('t', inplace=True)
    return (os.path.basename(filename), ts, fs, ts.columns)


def steval_dataset(filenames: List[str], fs: int) -> List[Tuple[str, pd.DataFrame]]:
    dataset = []
    for filename in sorted(filenames, key=lambda x: int(os.path.basename(x).split('.')[0])):
        name, ts, fs, cols = steval_measurement(filename, fs)
        dataset.append((filename, ts))
    return dataset


def beaglebone_measurement(filename: str, fs: int=2500) -> Tuple[str, pd.DataFrame]:
    g = 9.81
    milivolts = 1800
    resolution = 2 ** 12
    columns = ['x', 'y', 'z']
    ts = pd.read_csv(filename, delimiter='\t', index_col=False, header=None, names=columns)
        
    # Calculate amplitude in m/s^2 Beaglebone Black ADC and ADXL335 resolution (VIN 1.8V, 12bits)
    for dim in columns:
        ts[dim] = ts[dim] * (milivolts / resolution)  # ADC to mV
        ts[dim] = (ts[dim] / 180) * g                 # mV to m/s^2 (180 mV/g)
        ts[dim] -= ts[dim].mean()

    ts['t'] = ts.index * (1 / fs)
    ts.set_index('t', inplace=True)
    return (os.path.basename(filename), ts, fs, ts.columns)  # last is feature columns


def beaglebone_dataset(filenames: List[str]) -> List[Tuple[str, pd.DataFrame]]:
    dataset = []
    for filename in filenames:
        name, ts, fs, cols = beaglebone_measurement(filename)
        dataset.append((name, ts))
    return dataset


def import_files_split(file_list, func, loader, parts, cores=4):
    pool = ThreadPool(processes=cores)
    
    return pd.concat([
        pool.apply_async(func, (name, loader, parts)).get()
        for name in tqdm(file_list)
    ])
import os
from typing import (
    Callable,
    Tuple,
    List,
    Dict
)
from datetime import datetime
from zipfile import ZipFile

import numpy as np
import pandas as pd
from vibrodiagnostics import extraction


SAMPLING_RATE = 26866
"""Sampling frequency in Hz of the sensors
"""
BEARINGS_COLUMNS = ['x', 'y', 'z']
COLUMNS = ['t'] +  BEARINGS_COLUMNS
LABEL_COLUMNS = ['date', 'device', 'position']
"""Metadata columns extracted from file path witin dataset
"""
G_CONSTANT = 9.80665


def csv_import(dataset: ZipFile, filename: str) -> pd.DataFrame:
    """Open a CSV file from Pump zip archive

    :param dataset: ZIP archive of Pump industrial dataset
    :param filename: path to the file within dataset

    :return: data frame of the imported file
    """
    if dataset is not None:
        name = dataset.open(filename)
    else:
        name = filename

    ts = pd.read_csv(
        name,
        delimiter='\t',
        index_col=False,
        header=0,
        names=COLUMNS
    ) 
    columns = BEARINGS_COLUMNS
    ts[columns] = ts[columns].apply(lambda x: G_CONSTANT * (x / 1000))

    T = 1 / SAMPLING_RATE
    ts = (
        ts
        .assign(t = lambda x: x.index * T)
        .assign(key=filename)
    )
    return ts


def features_by_domain(
            features_calc: Callable,
            dataset: ZipFile,
            filename: str, 
            window: int = None,
            parts: int = None
        ) -> pd.DataFrame:
    """Open a CSV file from Pump zip archive

    :param features_calc: callback feature extraction function that
        has parameters for data frame, column to process in data frame,
        sampling frequency, window length of segment
    :param dataset: ZIP archive of Pump dataset
    :param filename: path to the file within dataset
    :param window: length of window (usually for FFT)
    :param parts: number of parts the input time series is split into

    :return: row(s) of features extracted from the file
    """

    fs = SAMPLING_RATE
    columns = BEARINGS_COLUMNS

    ts = csv_import(dataset, filename)
    dataframe = extraction.split_dataframe(ts, parts)
    dataframe = extraction.detrending_filter(dataframe, columns)
    
    header = filename.split(os.path.sep)
    metadata = [
        ('date', [datetime.fromisoformat(header[-4]).date()]),
        ('device', [header[-3]]),
        ('position', [header[-2]]),
        ('seq', [int(header[-1].split('.')[0])])
    ]

    result = []
    for i, df in enumerate(dataframe):
        fvector = metadata.copy()
        for col in columns:
            fvector.extend(features_calc(df, col, fs, window))
        result.append(pd.DataFrame(dict(fvector))) 

    return pd.concat(result).reset_index(drop=True)


def features_by_domain_no_metadata(
            features_calc: Callable,
            filename: str, 
            window: int = None,
            parts: int = None
        ) -> pd.DataFrame:

    fs = SAMPLING_RATE
    columns = BEARINGS_COLUMNS

    ts = csv_import(None, filename)
    dataframe = extraction.split_dataframe(ts, parts)
    dataframe = extraction.detrending_filter(dataframe, columns)

    result = []
    for i, df in enumerate(dataframe):
        fvector = []
        for col in columns:
            fvector.extend(features_calc(df, col, fs, window))
        result.append(pd.DataFrame(dict(fvector))) 
        
    X = pd.concat(result).reset_index(drop=True)
    feature_names = X.columns.str.extract(r'([a-z]+)_([a-z\_\-]+)')[1].unique()

    df = pd.DataFrame()
    for name in feature_names:
        vector_dims = [f'{dim}_{name}' for dim in columns]
        df[name] = X[vector_dims].apply(np.linalg.norm, axis=1)

    return df


def get_classes(df: pd.DataFrame, labels: Dict[str, dict], keep: bool = False) -> pd.DataFrame:
    """Assign labels to fault types into "label" column and optionally clean up data frame to
    contain only annotated rows with faults

    :param df: data frame after feature extraction with column "fault"
    :param labels: Llbels to assign machine and measurement placement
    :param keep: do not remove metadata columns

    :return: annotated data frame
    """
    df['label'] = df.apply(
        lambda row: labels.get(row['device'], {}).get(row['position']), axis=1
    )
    df['label'] = df['label'].astype('category')
    if not keep:
        df = df.dropna()
        df = df.drop(columns=LABEL_COLUMNS)
        df = df.reset_index(drop=True)
    
    return df


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign labels to fault types into "label" column based on meaurement placement

    :param df: data frame after feature extraction with column "fault"

    :return: annotated data frame
    """
    df['label'] = df['position']
    df['label'] = df['label'].astype('category')

    df = df.dropna()
    df = df.drop(columns=LABEL_COLUMNS)
    df = df.reset_index(drop=True)

    return df


def beaglebone_measurement(filename: str, fs: int) -> Tuple[str, pd.DataFrame]:
    """Import csv file recorded on BeagleBone Black with accelerometer ADXL335

    :param filename: file name of the recording
    :param fs: sampling frequency in Hz

    :return: data frame of the recording
    """
    milivolts = 1800
    resolution = 2**12
    columns = ['x', 'y', 'z']
    ts = pd.read_csv(filename, delimiter='\t', index_col=False, header=None, names=columns)
        
    # Calculate amplitude in m/s^2 Beaglebone Black ADC and ADXL335 resolution (VIN 1.8V, 12bits)
    for dim in columns:
        ts[dim] = ts[dim] * (milivolts / resolution)  # ADC to mV
        ts[dim] = (ts[dim] / 180) * G_CONSTANT        # mV to m/s^2 (180 mV/g)
        ts[dim] -= ts[dim].mean()

    ts['t'] = ts.index * (1 / fs)
    ts.set_index('t', inplace=True)
    return (os.path.basename(filename), ts, fs, ts.columns)  # last is feature columns
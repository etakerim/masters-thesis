import re
import os
from typing import (
    Callable,
    Tuple,
    List,
    Dict
)
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.signal import (
    find_peaks,
    butter,
    lfilter,
    welch
)
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from vibrodiagnostics import extraction


SAMPLING_RATE = 50000
"""Sampling frequency in Hz of the sensors
"""
BEARING_A_COLUMNS = ['ax', 'ay', 'az']
BEARING_B_COLUMNS = ['bx', 'by', 'bz']
BEARINGS_COLUMNS = BEARING_A_COLUMNS + BEARING_B_COLUMNS
COLUMNS = ['tachometer'] + BEARINGS_COLUMNS + ['mic']
LABEL_COLUMNS = ['fault', 'severity', 'rpm']
"""Metadata columns extracted from file path witin dataset
"""


BEARINGS = {
    'balls': 8,
    'ball_diameter': 0.7145,
    'pitch_diameter': 2.8519,
    'bpfo_factor': 2.9980,
    'bpfi_factor': 5.0020,
    'bsf_factor': 1.8710,
    'ftf_factor': 0.3750
}
"""Coeficients for bearing characteristic frequencies
"""


FAULTS = {
    'A': {
        'normal': 'normal',
        'imbalance': 'imbalance',
        'horizontal-misalignment': 'misalignment',
        'vertical-misalignment': 'misalignment',
        'underhang-outer_race': 'outer race fault',
        'underhang-cage_fault': 'cage fault',
        'underhang-ball_fault': 'ball fault'
    },
    'B': {
        'normal': 'normal',
        'imbalance': 'imbalance',
        'horizontal-misalignment': 'misalignment',
        'vertical-misalignment': 'misalignment',
        'overhang-outer_race': 'outer race fault',
        'overhang-cage_fault': 'cage fault',
        'overhang-ball_fault': 'ball fault',
    }
}
"""Annotation of fault types by bearing placement
"""


def bearing_frequencies(rpm: int) -> Dict[str, float]:
    """Calculate bearing characteristic frequencies for MaFaulDa machine simulator

    :param rpm: Rotational speed of the machine

    :return: Bearing defect frequencies
    """
    machine = {}
    rpm /= 60
    machine['RPM'] = rpm
    machine['BPFO'] = BEARINGS['bpfo_factor'] * rpm
    machine['BPFI'] = BEARINGS['bpfi_factor'] * rpm
    machine['BSF'] = BEARINGS['bsf_factor'] * rpm
    machine['FTF'] = BEARINGS['ftf_factor'] * rpm
    return machine


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """Split path of file within dataset structure to label the time series

    :param filename: path to file inside of zip archive

    :return: fault type, severity conditions, and file number
    """
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


def rpm_calc(tachometer: pd.Series) -> float:
    """Extract rotational speed in rpm units from tachometer pulse signal

    :param tachometer: tachometer signal

    :return: rotational speed in rpm
    """
    t = tachometer.index.to_numpy()
    y = tachometer.to_numpy()
    peaks, _ = find_peaks(y, prominence=3, width=50)
    interval = np.diff(t[peaks]).mean()
    return 60 / interval


def lowpass_filter(
        data: pd.Series,
        cutoff: int = SAMPLING_RATE // 5,
        fs: int = SAMPLING_RATE,
        order: int = 5) -> pd.Series:
    """Low-pass filter of n-th order the input signal at the cutoff frequency

    :param data: input signal
    :param cutoff: cutoff frequency
    :param fs: sampling frequency in Hz of the input signal
    :param order: steps of the filter

    :return: output signal after filtering
    """

    b, a = butter(order, cutoff, fs=fs, btype='lowpass')
    y = lfilter(b, a, data.to_numpy())
    return pd.Series(data=y, index=data.index)


def lowpass_filter_extract(
            dataframes: List[pd.DataFrame],
            columns: List[str]
        ) -> List[pd.DataFrame]:
    """Apply low-pass to columns in multiple data frames

    :param data frames: list of input datafrmaes to which filter is applied to
    :param columns: columns that filter is applied to

    :return: list of data frames after filtering
    """
    for df in dataframes:
        df[columns] = df[columns].apply(lowpass_filter)
    return dataframes


def csv_import(dataset: ZipFile, filename: str) -> pd.DataFrame:
    """Open a CSV file from MaFaulda zip archive

    :param dataset: ZIP archive of MaFaulDa dataset
    :param filename: path to the file within dataset

    :return: data frame of the imported file
    """
    columns = COLUMNS
    ts = pd.read_csv(dataset.open(filename), names=columns)
    T = 1 / SAMPLING_RATE
    ts = (
        ts
        .assign(t = lambda x: x.index * T)
        .set_index('t')
        .assign(rpm = lambda x: rpm_calc(x.tachometer))
    )
    ts[columns] = ts[columns].apply(lambda x: x - x.mean())
    ts[columns] = ts[columns].apply(lowpass_filter)
    return ts.assign(key=filename)


def features_by_domain(
        features_calc: Callable,
        dataset: ZipFile,
        filename: str,
        window: int = None,
        parts: int = 1,
        multirow: bool = False) -> pd.DataFrame:

    """Open a CSV file from MaFaulda zip archive

    :param features_calc: callback feature extraction function that
        has parameters for data frame, column to process in data frame,
        sampling frequency, window length of segment
    :param dataset: ZIP archive of MaFaulDa dataset
    :param filename: path to the file within dataset
    :param window: length of window (usually for FFT)
    :param parts: number of parts the input time series is split into
    :param multirow: extracted features are in rows, not in columns

    :return: row(s) of features extracted from the file
    """

    fs = SAMPLING_RATE
    columns = BEARINGS_COLUMNS

    ts = csv_import(dataset, filename)
    fault, severity, seq = parse_filename(filename)

    dataframe = extraction.split_dataframe(ts, parts)
    dataframe = extraction.detrending_filter(dataframe, columns)
    dataframe = lowpass_filter_extract(dataframe, columns)

    result = []
    for i, df in enumerate(dataframe):
        header = [
            ('fault', [fault]),
            ('severity', [severity]),
            ('seq', [f'{seq}.part.{i}']),
            ('rpm', [df['rpm'].mean()])
        ]
        for col in columns:
            if multirow is True:
                for data in features_calc(df, col, fs, window):
                    row = dict(header.copy())
                    row.update(data)
                    result.append(pd.DataFrame(row))
            else:
                header.extend(features_calc(df, col, fs, window))
                result.append(pd.DataFrame(dict(header)))

    return pd.concat(result).reset_index(drop=True)


def get_classes(df: pd.DataFrame, bearing: str) -> pd.DataFrame:
    """Create column "label" in data frame according to chosen bearing

    :param df: data frame after feature extraction with column "fault"
    :param bearing: bearing to determine labels of fault types ("A" or "B")

    :return: data frame with "label" column
    """
    if not bearing:
        faults = [f for f in FAULTS.values()]
        faults = {k: v for d in faults for k, v in d.items()}
        df['label'] = df.apply(lambda row: faults.get(row['fault']), axis=1)
    else:
        df['label'] = df.apply(lambda row: FAULTS[bearing].get(row['fault']), axis=1)
    df['label'] = df['label'].astype('category')
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove excessive columns with metadata and drop rows without label

    :param df: data frame with excess labels

    :return: data frame that consists of columns with features and "label"
    """
    df = df.dropna()
    df = df.drop(columns=LABEL_COLUMNS + ['severity_class', 'severity_level', 'severity_no'], errors='ignore')
    df = df.reset_index(drop=True)
    return df


def assign_labels(df: pd.DataFrame, bearing: str, keep: bool = False) -> pd.DataFrame:
    """Assign labels to fault types for bearing and optionally clean up data frame to
    contain only annotated rows with faults

    :param df: data frame after feature extraction with column "fault"
    :param bearing: bearing to determine labels of fault types ("A" or "B")
    :param keep: do not remove metadata columns

    :return: annotated data frame
    """
    df = get_classes(df, bearing)
    if not keep:
        df = clean_columns(df)
    return df


def mark_severity(df: pd.DataFrame, bearing: str, debug: bool = False) -> pd.DataFrame:
    """Calculate relative severity levels for data frame with original metadata columns

    :param df: data frame after feature extraction with column "fault" and "severity"
    :param bearing: bearing to determine labels of fault types ("A" or "B")
    :param debug: print relative severity levels

    :return: data frame with columns for absolute and relative fault severity levels
    """

    df = get_classes(df, bearing)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df['label'] = df['label'].astype('category')
    df['severity_no'] = df['severity'].str.extract(r'(\d+\.?\d*)').astype(float)

    for name, group in df.groupby(by=['label'], observed=True):
        group = group.sort_values(by='severity_no')

        severities = group['severity_no'].astype('category').cat.codes.values.reshape(-1, 1)
        scale_severities = MinMaxScaler().fit_transform(severities)

        df.loc[group.index, 'severity_class'] = severities
        df.loc[group.index, 'severity_level'] = scale_severities

        if debug is True:
            sev_names = list(group['severity'].astype('category').cat.categories)
            sev = list(group['severity'].astype('category').cat.codes.astype('category').cat.categories)
            scale = [
                float(f'{p:.2f}')
                for p in pd.Series(scale_severities[:, 0]).astype('category').cat.categories
            ]
            print(
                f'Fault: {name[0]}, Files: {len(group)}, '
                f'Severity names: {sev_names}, '
                f'Severity: {sev}, '
                f'Severity Levels: {scale}'
            )
    return df


def label_severity(
            df: pd.DataFrame,
            bearing: str,
            level: float,
            debug: bool = False,
            keep: bool = False
        ) -> pd.DataFrame:
    """Relabel faults less than set relative severity level as "normal"

    :param df: data frame after feature extraction with column "fault"
    :param bearing: bearing to determine labels of fault types ("A" or "B")
    :param debug: print relative severity levels
    :param keep: do not remove metadata columns

    :return: data frame with relabeled observations
    """
    df = mark_severity(df, bearing, debug)
    df.loc[df['severity_level'] < level, 'label'] = 'normal'
    if not keep:
        df = clean_columns(df)
    return df


def load_source(domain: str, row: dict, train_size: float = 0.8) -> tuple:
    """Load features according to domain and split the observations
    into training and testing set

    :param domain: complete feature set ("TD", "FD")
    :param row: parameters for data filtering, e.g.: {"placement": "A", online: False}
    :param train_size: ratio of traing set to testing set

    :return: X_train, X_test, Y_train, Y_test
    """
    PATH = '../datasets/'
    FEATURES_PATH = os.path.join(PATH, 'features')
    MAFAULDA_TEMPORAL = os.path.join(FEATURES_PATH, 'MAFAULDA_TD.csv')
    MAFAULDA_SPECTRAL = os.path.join(FEATURES_PATH, 'MAFAULDA_FD.csv')

    dataset = {
        'TD': MAFAULDA_TEMPORAL,
        'FD': MAFAULDA_SPECTRAL,
        'axis': {
            'A': ['ax', 'ay', 'az'],
            'B': ['bx', 'by', 'bz']
        },
        'labels': ['fault', 'severity', 'rpm']
    }

    placement = row['placement']
    df = extraction.load_features(
        dataset[domain],
        dataset['axis'][placement],
        dataset['labels']
    )
    frame = assign_labels(df, placement)
    Y = frame['label']
    X = frame.drop(columns=['label'])

    # Batch / Online hold-out (balance and event sequencing)
    if row.get('online') is True:
        features = label_severity(df, placement, 0.5, keep=True)
        # Shuffle order within severity level and order event with increasing severity
        groups = [
            frame.sample(frac=1, random_state=10)
            for i, frame in (
                features
                .sort_values(by='severity_level')
                .groupby('severity_level')
            )
        ]
        rows = list(pd.concat(groups).index)
        X = X.loc[rows].reset_index(drop=True)
        Y = Y.loc[rows].reset_index(drop=True)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, train_size=train_size, random_state=10
        )
        X_train, X_test, Y_train, Y_test = (
            X_train.sort_index(), X_test.sort_index(),
            Y_train.sort_index(), Y_test.sort_index()
        )

    else:
        oversample = RandomOverSampler(sampling_strategy='not majority', random_state=10)
        X, Y = oversample.fit_resample(X, Y.to_numpy())
        X.reset_index(drop=True, inplace=True)
        Y = pd.Series(Y)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, train_size=train_size, stratify=Y, random_state=10
        )

        scaler = MinMaxScaler()
        X_train[X_train.columns] = scaler.fit_transform(X_train)
        X_test[X_test.columns] = scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test

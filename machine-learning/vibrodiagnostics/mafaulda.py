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
BEARING_A_COLUMNS = ['ax', 'ay', 'az']
BEARING_B_COLUMNS = ['bx', 'by', 'bz']
BEARINGS_COLUMNS = BEARING_A_COLUMNS + BEARING_B_COLUMNS
COLUMNS = ['tachometer'] + BEARINGS_COLUMNS + ['mic']
LABEL_COLUMNS = ['fault', 'severity', 'rpm']

BEARINGS = {
    'balls': 8,
    'ball_diameter': 0.7145,
    'pitch_diameter': 2.8519,
    'bpfo_factor': 2.9980,
    'bpfi_factor': 5.0020,
    'bsf_factor': 1.8710,
    'ftf_factor': 0.3750
}

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


def bearing_frequencies(rpm: int) -> Dict[str, float]: 
    machine = {}
    rpm /= 60
    machine['RPM'] = rpm
    machine['BPFO'] = BEARINGS['bpfo_factor'] * rpm
    machine['BPFI'] = BEARINGS['bpfi_factor'] * rpm
    machine['BSF'] = BEARINGS['bsf_factor'] * rpm
    machine['FTF'] = BEARINGS['ftf_factor'] * rpm
    return machine


def parse_filename(filename: str) -> Tuple[str, str, str]:
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
    
    b, a = butter(order, cutoff, fs=fs, btype='lowpass')
    y = lfilter(b, a, data.to_numpy())
    return pd.Series(data=y, index=data.index)


def lowpass_filter_extract(dataframes: List[pd.DataFrame], columns: List[str]) -> List[pd.DataFrame]:
    for df in dataframes:
        df[columns] = df[columns].apply(lowpass_filter)
    return dataframes


def csv_import(dataset: ZipFile, filename: str) -> pd.DataFrame:
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
        parts: int = 1) -> pd.DataFrame:

    # print(f'Processing: {filename}')
    fs = SAMPLING_RATE
    columns = BEARINGS_COLUMNS

    ts = csv_import(dataset, filename)
    fault, severity, seq = parse_filename(filename)

    dataframe = extraction.split_dataframe(ts, parts)
    dataframe = extraction.detrending_filter(dataframe, columns)
    dataframe = lowpass_filter_extract(dataframe, columns)

    result = []
    for i, df in enumerate(dataframe):
        fvector = [
            ('fault', [fault]),
            ('severity', [severity]),
            ('seq', [f'{seq}.part.{i}']),
            ('rpm', [df['rpm'].mean()])
        ]
        for col in columns:
            fvector.extend(features_calc(df, col, fs, window))
        result.append(pd.DataFrame(dict(fvector))) 

    return pd.concat(result).reset_index(drop=True)


def get_classes(df: pd.DataFrame, bearing: str) -> pd.DataFrame:
    if not bearing:
        faults = [f for f in FAULTS.values()]
        faults = {k: v for d in faults for k, v in d.items()}
        df['label'] = df.apply(lambda row: faults.get(row['fault']), axis=1)
    else:
        df['label'] = df.apply(lambda row: FAULTS[bearing].get(row['fault']), axis=1)
    df['label'] = df['label'].astype('category')
    return df


def assign_labels(df: pd.DataFrame, bearing: str) -> pd.DataFrame:
    df = get_classes(df, bearing)
    df = df.dropna()
    df = df.drop(columns=LABEL_COLUMNS)
    df = df.reset_index(drop=True)
    return df


def label_severity(df: pd.DataFrame, bearing: str, level: float, debug: bool = False, keep: bool = False) -> pd.DataFrame:
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
            
    df.loc[df['severity_level'] < level, 'label'] = 'normal'
    if not keep:
        df = df.drop(columns=LABEL_COLUMNS)
        df = df.drop(columns=['severity_class', 'severity_level', 'severity_no'])
    return df


def load_source(domain: str, row: dict, train_size: float = 0.8) -> tuple:
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


def get_features_list(domain: str):
    PATH = '../datasets/'
    FEATURES_PATH = os.path.join(PATH, 'features')
    MAFAULDA_TEMPORAL = os.path.join(FEATURES_PATH, 'MAFAULDA_TD.csv')
    MAFAULDA_SPECTRAL = os.path.join(FEATURES_PATH, 'MAFAULDA_FD.csv')

    domains = {
        'TD': MAFAULDA_TEMPORAL,
        'FD': MAFAULDA_SPECTRAL,
    }
    features = {}
    for dname, dataset in domains.items():
        names = pd.read_csv(dataset)
        names = names.columns.str.extract(r'([a-z]{2})_([a-z\_\-]+)')[1].unique()
        features[dname] = names

    return features[domain]

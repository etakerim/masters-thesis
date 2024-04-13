import os.path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tsfel

from river import stats, preprocessing
from scipy.stats import pearsonr, pointbiserialr
from sklearn.feature_selection import mutual_info_classif, f_classif

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Dataset paths and attributes
MAFAULDA_METADATA = 'mafaulda_metadata.csv'
TIME_FEATURES_PATH = 'temporal_features.csv'
FREQ_FEATURES_PATH = 'spectral_features.csv'
TIME_AND_FREQ_FEATURES_PATH = 'time_freq_features.csv'


TIME_FEATURES_PATH_ALL = 'all_td_features.csv'
FREQ_FEATURES_PATH_ALL = 'all_fd_features.csv'
WPD_FEATURES_PATH_ALL = 'all_wpd_features.csv'
TSFEL_FEATURES_PATH_ALL = 'all_tsfel_features.csv'


WPD_COLUMNS_EXCLUDE = {
    'fault', 'severity', 'seq', 'rpm', 'axis', 'feature'
}

FAULT_CLASSES = {'normal': 'N', 'imbalance': 'I', 'horizontal-misalignment': 'HM', 'vertical-misalignment': 'VM'}
METADATA_COLUMNS = ['fault', 'severity', 'seq', 'rpm']
METADATA_COLUMNS_ALL = METADATA_COLUMNS + ['severity_class', 'severity_level', 'anomaly']

#########################################################
def filter_out_metadata_columns(df):
    return df[df.columns[~df.columns.isin(METADATA_COLUMNS_ALL)]]


def remove_column_prefix(df):
    return df.columns.str.extract(r'\w+_(\w+)')[0]


def fd_extract_feature_name(column):
    return column.str.extract(r'[A-Za-z]+_(.*)_\d+$')[0]


def plot_bar_chart(ax, x, y, title):
    ax.bar(x, y)
    ax.grid(True)
    ax.set_title(title)
    # Rotate x labels by 45 deg
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

def corr_features_to_fault(dataframe, features):
    fault_dummies = pd.get_dummies(dataframe['fault'])
    fault_features = pd.concat([dataframe, fault_dummies], axis=1)

    correlations = []
    for i, fault in enumerate(dataframe['fault'].cat.categories):
        for col in features:
            x = fault_features[col].fillna(0)
            y = fault_features[fault]
            f = np.abs(pearsonr(x, y)[0])
            c = {
                'fault': fault,
                'feature': col,
                'corr': f
            }
            correlations.append(c)

    correlations = pd.DataFrame(correlations)
    correlations['fault'] = correlations['fault'].astype('category')
    correlations['feature'] = correlations['feature'].astype('category')
    return correlations


def calc_corr_in_fft_windows(table):
    c = pd.DataFrame()
    window_sizes = (
        table.columns.str.extract(r'[\w\_]+_(\w+)$')[0]
             .dropna().unique().astype(int)
    )

    for win_len in window_sizes:
        columns = table.columns[table.columns.str.endswith(f'_{win_len}')]
        win_data = table.loc[:,
            table.columns.str.endswith(f'_{win_len}') | 
            table.columns.str.contains('fault')
        ]
        df = corr_features_to_fault(win_data, columns)
        df['window'] = win_len
        c = pd.concat([c, df])
    
    c['window'] = c['window'].astype('category')
    c['feature'] = c['feature'].astype('category')
    c = c.reset_index().drop(columns='index').drop_duplicates()
    return c


def calc_corr_in_wpd_features(corr_table):
    c = pd.DataFrame()

    for metric, group in corr_table.groupby(by='feature', observed=True):
        columns = list(set(group.columns) - WPD_COLUMNS_EXCLUDE)
        df = corr_features_to_fault(group, columns)
        df['metric'] = metric
        c = pd.concat([c, df])

    c['feature'] = c['feature'].astype('category')
    c['metric'] = c['metric'].astype('category')
    c = c.reset_index().drop(columns='index').drop_duplicates()
    return c


def weighted_rank_features_corr(features, index, weighted, values='corr'):
    df_ranks = pd.DataFrame()
    for i, group in enumerate(features.groupby(by='fault', observed=True)):
        fault, df = group
        corr_fault_to_feat = df.pivot(index=index, columns='feature', values=values)
        feature_ranks = corr_fault_to_feat.rank(axis='columns', method='dense', ascending=False)

        if weighted:
            feature_ranks *= corr_fault_to_feat
        
        common_rank = feature_ranks.mean().sort_values().to_frame(name='rank')
        
        common_rank['fault'] = fault
        df_ranks = pd.concat([df_ranks, common_rank])

    df_ranks['fault'] = df_ranks['fault'].astype('category')
    return df_ranks


def plot_ranked_features(ranks, n=None):
    num_of_faults = len(ranks['fault'].cat.categories)
    fig, ax = plt.subplots(2, num_of_faults // 2, figsize=(20, 10))

    for i, group in enumerate(ranks.groupby(by='fault', observed=True)):
        fault, df = group
        if n is not None:
            df = df.head(n)
        plot_bar_chart(ax.flatten()[i], df.index, df['rank'], f'Fault: {fault}') 

    fig.tight_layout()


#################################################################
############### F SCORE AND MUTUAL INFORMATION #################

def normalize_features(features, columns):
    standard_transformer = Pipeline(steps=[('standard', StandardScaler())])
    minmax_transformer = Pipeline(steps=[('minmax', MinMaxScaler())])
    preprocessor = ColumnTransformer(
        remainder='passthrough',
        transformers=[
            ('std', standard_transformer, columns)
        ],
        verbose_feature_names_out=False
    )
    features_normalized = preprocessor.fit_transform(features)
    features_normalized = pd.DataFrame(features_normalized, columns=preprocessor.get_feature_names_out())
    return features_normalized


def calc_feature_selection_metric(fmetric, dataset, columns):
    m = fmetric(dataset[columns], dataset['fault'])  # Do not have to be codes
    if isinstance(m, tuple):
        m = m[0]
    return (pd.DataFrame(list(zip(columns, m)), columns=['feature', 'stat'])
                .set_index('feature')
                .sort_values(by='stat', ascending=False))


def calc_corr_stat(dataset, columns):
    return calc_feature_selection_metric(corr_classif, dataset, columns)


def calc_f_stat(dataset, columns):
    return calc_feature_selection_metric(f_classif, dataset, columns)


def calc_mutual_information(dataset, columns):
    return calc_feature_selection_metric(mutual_info_classif, dataset, columns)


def calc_score_in_fft_windows(table, columns, func):
    c = pd.DataFrame()

    window_sizes = (
        table.columns.str.extract(r'[\w\_]+_(\w+)$')[0]
             .dropna().unique().astype(int)
    )

    for win_len in window_sizes:
        columns = table.columns[table.columns.str.endswith(f'_{win_len}')]
        win_data = table.loc[:,
            table.columns.str.endswith(f'_{win_len}') | 
            table.columns.str.contains('fault')
        ]
        df = func(win_data, columns)
        df['window'] = win_len
        c = pd.concat([c, df])
    
    c['window'] = c['window'].astype('category')
    c.reset_index(inplace=True)
    c['feature'] = fd_extract_feature_name(c['feature'])
    return c


def calc_score_in_wpd_features(src, func):
    c = pd.DataFrame()

    for metric, group in src.groupby(by='feature', observed=True):
        columns = list(set(group.columns) - WPD_COLUMNS_EXCLUDE)
        df = func(group, columns)
        df['metric'] = metric
        c = pd.concat([c, df])

    c['metric'] = c['metric'].astype('category')
    return c


def plot_fscore_part(df, part, title, n=None):
    num_of_windows = len(df[part].cat.categories)
    fig, ax = plt.subplots(1, num_of_windows, figsize=(20, 4))

    for i, grouper in enumerate(df.groupby(by=part, observed=True)):
        h, group = grouper
        if n is not None:
            group = group.iloc[:n]
        group.plot.bar(grid=True, xlabel='Feature', ylabel=title, legend=False, title=h, ax=ax[i])

    fig.tight_layout()
    plt.show()


def plot_fscore_in_fft_win(df, title):
    num_of_windows = len(df['window'].cat.categories)
    fig, ax = plt.subplots(1, num_of_windows, figsize=(20, 4))
    for i, grouper in enumerate(df.groupby(by='window', observed=True)):
        win, group = grouper
        group.plot.bar(grid=True, xlabel='Feature', ylabel=title, legend=False, title=f'Window = {win}', ax=ax[i])
    fig.tight_layout()
    plt.show()

#### RANKING

def weighted_rank_features(df, index, weighted=False):
    score_to_feat = df.reset_index().pivot(index=index, columns='feature', values='stat')
    feature_ranks = score_to_feat.rank(axis='columns', method='dense', ascending=False)

    if weighted:
        feature_ranks *= score_to_feat
    rank = feature_ranks.mean().sort_values().to_frame(name='rank')
    return rank


def plot_rank(df, index):
    rank_non_weighted = weighted_rank_features(df, index, weighted=False)
    rank_weighted = weighted_rank_features(df, index, weighted=True)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    rank_non_weighted.plot.bar(grid=True, xlabel='Feature', ylabel='Rank', legend=False, title='Non-weighted', ax=ax[0])
    rank_weighted.plot.bar(grid=True, xlabel='Feature', ylabel='Rank weighted', legend=False, title='Weighted', ax=ax[1])
    plt.show()

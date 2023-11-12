import os.path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from river import stats, preprocessing
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif, f_classif

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Dataset paths and attributes
TIME_FEATURES_PATH = 'td_features.csv'
FREQ_FEATURES_PATH = 'fd_features.csv'
TIME_AND_FREQ_FEATURES_PATH = 'time_freq_features.csv'

TSFEL_FEATURES_PATH = 'tsfel_features.csv'
TSFEL_TIME_FEATURES_PATH = 'tsfel_td_features.csv'
TSFEL_FREQ_FEATURES_PATH = 'tsfel_fd_features.csv'
TSFEL_STAT_FEATURES_PATH = 'tsfel_sd_features.csv'

WPD_FEATURES_PATH = 'wpd_features.csv'

TIME_FEATURES_PATH_ALL = 'all_td_features.csv'
FREQ_FEATURES_PATH_ALL = 'all_fd_features.csv'
WPD_FEATURES_PATH_ALL = 'all_wpd_features.csv'
TSFEL_FEATURES_PATH_ALL = 'all_tsfel_features.csv'

TD_COLUMNS = ['mean', 'std', 'skew', 'kurt', 'rms', 'pp', 'crest', 'margin', 'impulse', 'shape']
FD_COLUMNS = [
    'centroid', 'std', 'skew', 'kurt', 'roll-off', 'flux_mean', 'flux_std',
    'noisiness', 'energy', 'entropy', 'negentropy'
]
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


def merge_feature_domains(time_domain: str, freq_domain: str):   
    td_features = pd.read_csv(time_domain)
    fd_features = pd.read_csv(freq_domain)
    merged = td_features.merge(
        fd_features, on=['fault', 'severity', 'seq', 'rpm'], 
        how='inner', validate='one_to_one'
    )
    return merged


def load_td_feat(axis, path='', all=False):
    filename = TIME_FEATURES_PATH_ALL if all else TIME_FEATURES_PATH

    features = pd.read_csv(os.path.join(path, filename))
    columns = features.columns.isin(METADATA_COLUMNS) | features.columns.str.startswith(tuple(axis))
    features = features[features.columns[columns]]
    features['fault'] = features['fault'].astype('category')
    return features


def load_fd_feat(axis, path='', all=False):
    filename = FREQ_FEATURES_PATH_ALL if all else FREQ_FEATURES_PATH

    features = pd.read_csv(os.path.join(path, filename))
    columns = features.columns.isin(METADATA_COLUMNS) | features.columns.str.startswith(tuple(axis))
    features = features[features.columns[columns]]
    features['fault'] = features['fault'].astype('category')
    return features


def load_wavelet_domain_features(axis, path='', all=False):
    filename = WPD_FEATURES_PATH_ALL if all else WPD_FEATURES_PATH

    features = pd.read_csv(os.path.join(path, filename))
    features['axis'] = features['axis'].astype('category')
    features['feature'] = features['feature'].astype('category')
    features['fault'] = features['fault'].astype('category')
    features = features[features['axis'].isin(axis)]
    return features


def plot_bar_chart(ax, x, y, title):
    ax.bar(x, y)
    ax.grid(True)
    ax.set_title(title)
    # Rotate x labels by 45 deg
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


############### ONLINE FISHER SCORE ###############################
class FisherScore(stats.base.Bivariate):

    def __init__(self):
        self.mean_x = stats.Mean()
        self.labels = {}

    def update(self, x, y):
        self.mean_x.update(x)

        if y not in self.labels:
            self.labels[y] = {
                'n': stats.Sum(), 
                'mean': stats.Mean(), 
                'var': stats.Var()
            }

        for s in self.labels[y].values():
            s.update(x)
        
        return self

    def get(self):
        mean_x = self.mean_x.get()

        inter_class = sum([
            s['n'].get() * ((s['mean'].get() - mean_x) ** 2)
            for s in self.labels.values()
        ])
        intra_class = sum([
            (s['n'].get() - 1) * s['var'].get()
            for s in self.labels.values()
        ])
        if intra_class == 0:
            return 0

        score = inter_class / intra_class
        return score


class MutualInformation(stats.base.Bivariate):

    def __init__(self):
        self.n = stats.Count()
        self.scaler = preprocessing.MinMaxScaler()
        self.label_bins = 3
        self.labels = {}

        self.matrix = []
        self.row_margins = []
        self.col_margins = [
            stats.Count() for i in range(self.label_bins)
        ]

    def update(self, x, y):
        self.n.update()
        if y not in self.labels:
            self.labels[y] = len(self.matrix)
            self.row_margins.append(stats.Count())
            self.matrix.append([
                stats.Count() for i in range(self.label_bins)
            ])
        
        row = self.labels[y]
        col = self.scaler.learn_one({'x': x}).transform_one({'x': x})
        col = int(col['x'] * self.label_bins)
        col = col - 1 if col >= self.label_bins else col

        self.row_margins[row].update()
        self.col_margins[col].update()
        self.matrix[row][col].update()
        return self

    def get(self):
        n = self.n.get()
        score = 0
        for r in range(len(self.matrix)):
            for c in range(len(self.matrix[r])):
                p_xy = self.matrix[r][c].get() / n
                p_x = self.row_margins[r].get() / n
                p_y = self.col_margins[c].get() / n
                if p_x != 0 and p_y != 0 and p_xy != 0:
                    score += p_xy * math.log(p_xy / (p_x * p_y))

        return score


#################################################################
##################### CORRELATION ###############################

def corr_classif(X, y):
    X = pd.DataFrame(X)
    y_dummies = pd.get_dummies(y)
    dataset = pd.concat([X, y_dummies], axis=1)
    scores = []

    for col in X.columns:
        x = dataset[col].fillna(0)
        corr = np.array([
            np.abs(pearsonr(x, dataset[category])[0])
            for category in np.unique(y)
        ])
        scores.append(corr.mean())

    return np.array(scores)


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

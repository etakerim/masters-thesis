import pandas as pd


# Dataset paths and attributes
TIME_FEATURES_PATH = '../../../datasets/features_data/td_features_no_filter.csv'
FREQ_FEATURES_PATH = '../../../datasets/features_data/fd_features_no_filter.csv'
WPD_FEATURES_PATH = '../../../datasets/features_data/wpd_features_no_filter.csv'

TD_COLUMNS = ['mean', 'std', 'skew', 'kurt', 'rms', 'pp', 'crest', 'margin', 'impulse', 'shape']
FD_COLUMNS = [
    'centroid', 'std', 'skew', 'kurt', 'roll-off', 'flux_mean', 'flux_std',
    'noisiness', 'energy', 'entropy', 'negentropy'
]
WPD_COLUMNS_EXCLUDE = {
    'fault', 'severity', 'seq', 'rpm', 'axis', 'feature'
}


def load_time_domain_features(axis):
    features = pd.read_csv(TIME_FEATURES_PATH)
    features = features[features['axis'].isin(axis)]
    features['fault'] = features['fault'].astype('category')
    return features


def load_frequency_domain_features():
    features = pd.read_csv(FREQ_FEATURES_PATH)
    features['fault'] = features['fault'].astype('category')
    features['fft_window_length'] = features['fft_window_length'].astype('category')
    return features


def load_wavelet_domain_features(axis):
    features = pd.read_csv(WPD_FEATURES_PATH)
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
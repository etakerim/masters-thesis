import numpy as np
import pandas as pd
from vibrodiagnostics.selection import  METADATA_COLUMNS_ALL
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.feature_selection import SelectPercentile, SelectKBest


def features_subset(filename, classes, axis, label=None, severity_sort=False, anomaly_severity=0.5):
    features = pd.read_csv(filename)
    features = fault_labeling(features, classes, anomaly_severity=anomaly_severity, debug=False)

    if severity_sort:
        groups = [
            df.sample(frac=1, random_state=10)
            for i, df in (
                features.sort_values(by='severity_level')
                        .groupby('severity_level')
            )
        ]
        features = pd.concat(groups).reset_index(drop=True)

    else:
        features = (         # Shuffle
            features
            .sample(frac=1, random_state=10)
            .reset_index(drop=True)
        )

    columns = features.columns.str.startswith(tuple(axis))
    features.columns[columns]
    X = features[features.columns[columns]]
    if label is None:
        return X
    else:
        y = features[label].astype('category').cat.codes
        return X, y


def features_subset_offline(filename, classes, axis, label, train_size=0.7, anomaly_severity=0.5):
    features = pd.read_csv(filename)
    features = fault_labeling(features, classes, anomaly_severity=anomaly_severity, debug=False)

    columns = features.columns.str.startswith(tuple(axis))
    features.columns[columns]
    X = features[features.columns[columns]]
    y = features[label].astype('category')

    if label == 'anomaly':
        X, y = anomalies_undersample(X, y, 0.1)

    return train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=10
    )


def fault_labeling(dataset, classes, anomaly_severity=0.7, debug=False):
    # Faults
    df = dataset.copy()
    df['fault'] = df['fault'].astype('category')
    df['fault'] = df['fault'].cat.rename_categories(classes)
    # Print classes of faults
    if debug is True:
        print('Faults:', list(df['fault'].cat.categories), end='\n\n')
    
    # Number fault severities by sequence
    df['seq'] = (
        df.groupby(by=['fault', 'severity'], observed=True)
             .cumcount().astype(int)
    )
    # Keep only decimal numbers in severity
    df['severity'] = df['severity'].str.extract(r'(\d+\.?\d*)').astype(float)

    # Number severity per group (0 - best, 1 - worst)
    for name, group in df.groupby(by=['fault'], observed=True):
        group = group.sort_values(by='severity')
            
        severities = group['severity'].astype('category').cat.codes.values.reshape(-1, 1)
        # Transorm to range (0, 1)
        scale_severities = MinMaxScaler().fit_transform(severities)
        
        df.loc[group.index, 'severity_class'] = severities
        df.loc[group.index, 'severity_level'] = scale_severities

        if debug is True:
            # Print severity scales
            sev_names = list(group['severity'].astype('category').cat.categories)
            sev = list(group['severity'].astype('category').cat.codes.astype('category').cat.categories)
            scale = [float(f'{p:.2f}') for p in pd.Series(scale_severities[:, 0]).astype('category').cat.categories]
            print(f'Fault: {name[0]}, Files: {len(group)}, Severity names: {sev_names}, Severity: {sev}, Severity Levels: {scale}')

    df['anomaly'] = (df['severity_level'] >= anomaly_severity)
    df['anomaly'] = df['anomaly'].astype('category')
    return df


def highly_correlated_features(df, corr=0.95):
    # https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find features with correlation greater than "corr"
    to_drop = [column for column in upper.columns if any(upper[column] > corr)]
    return to_drop


def filter_out_metadata_columns(df):
    return df[df.columns[~df.columns.isin(METADATA_COLUMNS_ALL)]]


def anomalies_undersample(X, y, anomaly_ratio):
    majority_class = y[y == False]
    minority_class = y[y == True]

    majority = len(majority_class)
    minority = int((anomaly_ratio * majority) / (1 - anomaly_ratio))

    # Undersample majority class when not enough anomalies is in the dataset
    if minority > len(minority_class):
        minority = len(minority)
        majority = int((minority - anomaly_ratio * minority) / anomaly_ratio)

    y = pd.concat([
        majority_class.sample(n=majority, random_state=100), 
        minority_class.sample(n=minority, random_state=100)
    ])
    X = X.iloc[y.index]
    return X, y


def pipeline_v1_core(func_select, nfeat, X_train, y_train, X_test, y_test):
    # Drop colinear features
    X_train = X_train.copy()
    X_test = X_test.copy()
    to_drop = highly_correlated_features(X_train)
    X_train.drop(to_drop, axis=1, inplace=True)
    X_test.drop(to_drop, axis=1, inplace=True)
    
    # Feature selection
    if nfeat >= len(X_train.columns):
        nfeat = 'all'

    # selector = SelectPercentile(func_select, percentile=20)
    selector = SelectKBest(func_select, k=nfeat)
    
    selector.fit_transform(X_train, y_train)
    selector.transform(X_test)
    idx = selector.get_support(indices=True)
    X_train = X_train.iloc[:,idx]
    X_test = X_test.iloc[:,idx]
       
    # Normalize features (See inverse transform)
    scaler = MinMaxScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train)
    X_test[X_test.columns] = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def pipeline_v1(features, train, nfeat, func_select, multiclass=True, anomaly_ratio=0.1):
    # Split features dataset to training and testing sets
    X = filter_out_metadata_columns(features)

    if multiclass is True:
        y = features['fault']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train, stratify=y, random_state=100
        )
    else:
        y = features['anomaly']
        X, y = anomalies_undersample(X, y, anomaly_ratio)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train, random_state=100
        )

    return pipeline_v1_core(
        func_select, nfeat,
        X_train, y_train, X_test, y_test
    )


def cross_cuts_3d(X_train, y_train):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3))
    for i, axes in enumerate(((0, 1), (0, 2), (1, 2))):
        a, b = axes

        for label, color in (('VM', 'purple'), ('N', 'green'), ('I', 'blue'), ('HM', 'orange')):
            x = X_train.loc[
                list(y_train[y_train == label].index), 
                X_train.columns[a]
            ]
            y = X_train.loc[
                list(y_train[y_train == label].index),
                X_train.columns[b]
            ]
            ax[i].scatter(x, y, s=1, color=color, label=label)
        
        ax[i].set_xlabel(X_train.columns[a])
        ax[i].set_ylabel(X_train.columns[b])
        ax[i].grid()
        ax[i].legend()


def cross_cuts_3d_anomalies(dataframe, anomalies):
    df = dataframe.copy()
    df['anomaly'] = anomalies
    df['anomaly'] = df['anomaly'].astype('category')
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 3))
    
    for i, axes in enumerate(((0, 1), (0, 2), (1, 2))):
        a, b = axes
        ax[i].grid()
        x = df.loc[:, df.columns[a]]
        y = df.loc[:, df.columns[b]]
        ax[i].scatter(x, y, color='grey', s=1)

        for flag, color in ((False, 'green'), (True, 'red')):
            points = list(df[df['anomaly'] == flag].index)
            x = df.loc[points, df.columns[a]]
            y = df.loc[points, df.columns[b]]
            ax[i].scatter(x, y, color=color, s=1)


def scatter_classif(X, y_label, categories, colors, ax):
    for label, color in zip(categories, colors):
            rows = list(y_label[y_label == label].index)
            x = X.loc[rows,0]
            y = X.loc[rows,1]
            ax.scatter(x, y, s=2, color=color, label=label)


def project_classifier_map_plot(X, y_true, y_predict): 
    y_true = y_true.astype('category') 
    y_predict = y_predict.astype('category')

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca)

    categories = y_true.cat.categories
    colors = sb.color_palette('hls', len(categories))

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    scatter_classif(X_pca, y_true, categories, colors, ax[0])
    scatter_classif(X_pca, y_predict, categories, colors, ax[1])

    match = y_predict == y_true[y_predict.index]
    good = y_predict[match == True].index
    bad = y_predict[match == False].index

    ax[2].scatter(X_pca[0].loc[good], X_pca[1].loc[good], s=2, color='green', label='Good')
    ax[2].scatter(X_pca[0].loc[bad], X_pca[1].loc[bad], s=2, color='red', label='Bad')
    
    var = 100 * pca.explained_variance_ratio_
    for i in range(3):
        ax[i].set_xlabel(f'PC1 ({var[0]:.2f} %)')
        ax[i].set_ylabel(f'PC2 ({var[1]:.2f} %)')
        ax[i].grid()
        ax[i].legend()

    return bad


def project_anomaly_map_plot(X, y_true, y_score, threshold=7):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca)

    y_true = y_true.astype('category')
    categories = y_true.cat.categories
    colors = sb.color_palette('hls', len(categories))

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    scatter_classif(X_pca, y_true, categories, colors, ax[0])
    anomaly_colors = sb.color_palette('Blues', 10).as_hex()
    
    scaler = MinMaxScaler(feature_range=(0, 9))
    scores = scaler.fit_transform(y_score.reshape(-1, 1))
    scores = scores.reshape(1, -1)[0]

    for x, y, score in zip(X_pca[0], X_pca[1], scores):
        ax[1].plot(x, y, '.', color=anomaly_colors[int(score)], markersize=2)

    for x, y, score in zip(X_pca[0], X_pca[1], scores):
        color = "red" if int(score) >= threshold else "green"
        ax[2].plot(x, y, '.', color=color, markersize=2)
    
    for i in range(3):
        ax[i].set_xlabel(f'PC1')
        ax[i].set_ylabel(f'PC2')
        ax[i].grid()
        ax[i].legend()
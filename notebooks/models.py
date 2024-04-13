from typing import (
    Dict,
    Tuple,
    List
)
import itertools
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

from tqdm.notebook import tqdm
import ranking


def transform_to_pca(X, n):
    scaler = MinMaxScaler()
    X[X.columns] = scaler.fit_transform(X)

    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca)
    return X_pca


def find_best_subset(
        X: pd.DataFrame,
        Y: pd.DataFrame,
        metric: str,
        members: int = 3,
        kfolds: int = 5
    ):
    Y = Y.dropna().astype('category')
    X = X[X.index.isin(Y.index)].copy()
    Y = Y[Y.index.isin(X.index)].astype('category')
    X = X.reset_index(drop=True)
    Y = Y.reset_index(drop=True)

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=10)
    elements = []

    for train_idx, test_idx in kf.split(X, Y):
        x_train, x_test, y_train, y_test = (
            X.loc[train_idx].copy(), X.loc[test_idx].copy(),
            Y.loc[train_idx].copy(), Y.loc[test_idx].copy()
        )
        ranks = ranking.batch_feature_ranking(x_train, y_train, metric)
        if metric != 'rank':
            synonyms = ranking.compute_correlations(x_train, corr_above=0.95)
            subset = ranking.best_subset(ranks, synonyms, n=members)
            output = subset
        else:
            output = ranks

        output = list(output.reset_index().head(3)['feature'])

    return output


def kfold_accuracy(
        X: pd.DataFrame,
        Y: pd.DataFrame,
        k_neighbors: int,
        kfolds: int,
        model_name: str, 
        knn_metric='euclidean') -> Dict[str, float]:

    # Remove missing data
    Y = Y.dropna()
    X = X[X.index.isin(Y.index)]
    Y = Y[Y.index.isin(X.index)].astype('category')

    # Class balancing
    oversample = RandomOverSampler(sampling_strategy='not majority', random_state=10)
    X, Y = oversample.fit_resample(X, Y.to_numpy())
    X.reset_index(drop=True, inplace=True)
    Y = pd.Series(Y)

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=10)
    round_train_acc = []
    round_test_acc = []

    for train_idx, test_idx in kf.split(X, Y):
        # Train / Test split in KFold
        x_train, x_test, y_train, y_test = (
            X.loc[train_idx].copy(), X.loc[test_idx].copy(),
            Y.loc[train_idx].copy(), Y.loc[test_idx].copy()
        )
        # Scale
        scaler = MinMaxScaler()
        x_train[x_train.columns] = scaler.fit_transform(x_train)
        x_test[x_test.columns] = scaler.transform(x_test)
    
        # Train k-NN model on all features
        if model_name == 'knn':
            model = KNeighborsClassifier(n_neighbors=k_neighbors, metric=knn_metric)   #, algorithm='kd_tree')
        elif model_name == 'lda':
            model = LinearDiscriminantAnalysis()
        elif model_name == 'bayes':
            model = GaussianNB()
        elif model_name == 'svm':
            model = LinearSVC()

        model.fit(x_train, y_train)
        y_predict_train = model.predict(x_train)
        y_predict_test = model.predict(x_test)

        round_train_acc.append(metrics.accuracy_score(y_train, y_predict_train))
        round_test_acc.append(metrics.accuracy_score(y_test, y_predict_test))
    
    return {
        'train': np.array(round_train_acc).mean(),
        'test': np.array(round_test_acc).mean()
    }


def all_features(
        X: pd.DataFrame,
        Y: pd.DataFrame,
        model: str,
        k_neighbors: list = list(range(1, 40, 4)),
        kfold_param: int = 5) -> dict:
    # Remove missing data
    Y = Y.dropna()
    X = X[X.index.isin(Y.index)]
    Y = Y[Y.index.isin(X.index)].astype('category')

    # Class balancing
    oversample = RandomOverSampler(sampling_strategy='not majority', random_state=10)
    X, Y = oversample.fit_resample(X, Y.to_numpy())
    X.reset_index(drop=True, inplace=True)
    Y = pd.Series(Y)

    train_accuracy = []
    test_accuracy = []

    for k in k_neighbors:
        accuracy = kfold_accuracy(X, Y, k, kfold_param, model)
        train_accuracy.append(accuracy['train'])
        test_accuracy.append(accuracy['test'])

    return {
        'k': k_neighbors,
        'train': train_accuracy,
        'test': test_accuracy
    }


def feature_combinations(
        X: pd.DataFrame,
        Y: pd.DataFrame,
        k_neighbors: int,
        num_of_features: int,
        kfolds: int,
        domain: str,
        model: str) -> List[dict]:
    
    results = []
    for features in tqdm(itertools.combinations(X.columns, r=num_of_features)):
        r = kfold_accuracy(X[list(features)], Y, k_neighbors, kfolds, model)
        r.update({'features': list(features), 'f': num_of_features, 'k': k_neighbors, 'domain': domain})
        results.append(r)
    return results


def enumerate_models(
        X_temporal: pd.DataFrame,
        X_spectral: pd.DataFrame,
        Y: pd.DataFrame,
        k_neighbors: Tuple[int] = (3, 5, 11, 15),
        num_of_features: Tuple[int] = (2, 3, 4, 5), 
        kfolds=5,
        model='knn') -> pd.DataFrame:

    models = []
    domains = {'temporal': X_temporal, 'spectral': X_spectral}

    for fnum in num_of_features:
        for domain, X in domains.items():
            for k in k_neighbors:
                result = feature_combinations(X, Y, k, fnum, kfolds, domain, model)
                models.extend(result)
                
    return pd.DataFrame.from_records(models)


def model_boundaries(X, Y, n=5, model_name='knn', knn_metric='euclidean'):
    # Class balancing
    oversample = RandomOverSampler(sampling_strategy='not majority', random_state=10)
    X, Y = oversample.fit_resample(X, Y.to_numpy())
    X.reset_index(drop=True, inplace=True)
    Y = pd.Series(Y)

    kf = KFold(n_splits=5, shuffle=True, random_state=10)

    for train_idx, test_idx in kf.split(X, Y):
        # Train / Test split in KFold
        x_train, x_test, y_train, y_test = (
            X.loc[train_idx].copy(), X.loc[test_idx].copy(),
            Y.loc[train_idx].copy(), Y.loc[test_idx].copy()
        )
        break

    if model_name == 'knn':
        model = KNeighborsClassifier(n_neighbors=n, metric=knn_metric)
    elif model_name == 'lda':
        model = LinearDiscriminantAnalysis()
    elif model_name == 'bayes':
        model = GaussianNB()
    elif model_name == 'svm':
        model = LinearSVC()

    model.fit(x_train, y_train)

    return model
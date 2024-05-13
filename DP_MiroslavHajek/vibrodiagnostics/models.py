from typing import (
    Dict,
    Tuple,
    List
)
import itertools
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import percentileofscore

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

from tqdm.notebook import tqdm
from vibrodiagnostics import ranking

from sklearn import metrics as skmetrics
from tqdm.notebook import tqdm
from river import metrics
from river import preprocessing
from river import neighbors, utils, evaluate, stream


def transform_to_pca(X: pd.DataFrame, n: int) -> pd.DataFrame:
    """Transform features to their principal components after normalization

    :param dataset: data frame with columns only for predictors
    :param n: number of principal components
    :return: data frames with rows of features replaced for principal components
    """
    scaler = MinMaxScaler()
    X[X.columns] = scaler.fit_transform(X)

    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca)
    return X_pca


def find_best_subset(
            X: pd.DataFrame,
            Y: pd.Series,
            metric: str,
            members: int,
            kfolds: int = 5
        ) -> List[str]:
    """Find the best subset of features based on supplied feature selection metric name

    :param X: data frame of predictor features
    :param Y: column of labels
    :param metric: name of the bivariate similarity metric to compute for each feature and label
    :param members: number of features in the subset
    :param kfolds: number of splits for k-fold cross validation
    :return: list of the best features according to feature selection metric
    """

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=10)
    elements = []

    train_idx, test_idx = next(kf.split(X, Y))
    x_train, x_test, y_train, y_test = (
        X.loc[train_idx].copy(), X.loc[test_idx].copy(),
        Y.loc[train_idx].copy(), Y.loc[test_idx].copy()
    )
    ranks = ranking.batch_feature_ranking(x_train, y_train, metric)
    # synonyms = ranking.compute_correlations(x_train, corr_above=0.95)
    synonyms = set()
    subset = ranking.best_subset(ranks, synonyms, n=members)
    subset = subset[subset['rank'] == True].index.to_list()
    return subset


def kfold_accuracy(
            X: pd.DataFrame,
            Y: pd.Series,
            k_neighbors: int,
            kfolds: int,
            model_name: str = 'knn',
            power_transform: bool = True,
            knn_metric='euclidean'
        ) -> Dict[str, float]:

    """Evaluate classifier accuracy in k-fold validation on a data frame of features after
    oversampling to the majority class

    :param X: data frame of predictor features
    :param Y: column of labels
    :param k_neighbors: number of neighbours for k-nearest neighbours model
    :param model_name: name of the machine learning model to evaluate.
        Options are: "knn", "lda", "bayes", "svm"
    :param kfolds: number of splits for k-fold cross-validation
    :param power_transform: apply power transform of features in preprocessing instead
        of normalization
    :param knn_metric: distance metric name for k-nearest neighbours model

    :return: average accuracy of model over k-folds in training and testing sets
    """

    # Class balancing
    oversample = RandomOverSampler(sampling_strategy='not majority', random_state=10)
    X, Y = oversample.fit_resample(X, Y.to_numpy())
    X = X.reset_index(drop=True)
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

        if power_transform is True:
            transform = PowerTransformer(method='yeo-johnson', standardize=True)
        else:
            transform = MinMaxScaler()

        x_train[x_train.columns] = transform.fit_transform(x_train)
        x_test[x_test.columns] = transform.transform(x_test)

        # Train k-NN model on all features
        if model_name == 'knn':
            model = KNeighborsClassifier(n_neighbors=k_neighbors, metric=knn_metric)
        elif model_name == 'lda':
            model = LinearDiscriminantAnalysis()
        elif model_name == 'bayes':
            model = GaussianNB()
        elif model_name == 'svm':
            model = LinearSVC()

        model.fit(x_train, y_train)
        y_predict_train = model.predict(x_train)
        y_predict_test = model.predict(x_test)

        round_train_acc.append(skmetrics.accuracy_score(y_train, y_predict_train))
        round_test_acc.append(skmetrics.accuracy_score(y_test, y_predict_test))

    return {
        'train': np.array(round_train_acc).mean(),
        'test': np.array(round_test_acc).mean()
    }


def all_features(
            X: pd.DataFrame,
            Y: pd.Series,
            model_name: str = 'knn',
            power_transform: bool = False,
            k_neighbors: list = list(range(1, 40, 4)),
            kfolds: int = 5
        ) -> Dict[str, float]:
    """Evaluate complete feature sets in k-nearest neighbours classifier
        with various k-value parameter

    :param X: data frame of predictor features
    :param Y: column of labels
    :param model_name: name of the machine learning model to evaluate.
        Options are: "knn", "lda", "bayes", "svm"
    :param power_transform: apply power transform of features in preprocessing instead
        of normalization
    :param k_neighbors: number of neighbours for k-nearest neighbours model
    :param kfolds: number of splits for k-fold cross-validation

    :return: training and testing accuracy for each value of k-neighbours
    """
    train_accuracy = []
    test_accuracy = []

    for k in k_neighbors:
        accuracy = kfold_accuracy(
            X, Y, k, kfolds, model_name,
            power_transform=power_transform
        )
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
            model: str,
            power_transform: bool = False
        ) -> List[dict]:
    """Evaluate all combinations of feature subsets of given size out of complete sets

    :param X: data frame of predictor features
    :param Y: column of labels
    :param k_neighbors: number of neighbours for k-nearest neighbours model
    :param num_of_features: number of features in the subset
    :param kfolds: number of splits for k-fold cross-validation
    :param domain: source domain from which the features are extracted
    :param model_: name of the machine learning model to evaluate.
        Options are: "knn", "lda", "bayes", "svm"
    :param power_transform: apply power transform of features in preprocessing instead
        of normalization

    :return: training and testing accuracy for each combination of features
    """
    results = []
    for features in tqdm(list(itertools.combinations(X.columns, r=num_of_features))):
        r = kfold_accuracy(
            X[list(features)], Y, k_neighbors, kfolds, model,
            power_transform=power_transform
        )
        r.update({
            'features': list(features),
            'f': num_of_features,
            'k': k_neighbors,
            'domain': domain
        })
        results.append(r)
    return results


def enumerate_models(
            X: pd.DataFrame,
            Y: pd.DataFrame,
            domain: str,
            k_neighbors: Tuple[int] = (3, 5, 11, 15),
            num_of_features: Tuple[int] = (2, 3, 4, 5),
            kfolds: int =5,
            power_transform: bool = False,
            model='knn'
        ) -> pd.DataFrame:
    """Grid search of parameters k-nearest neighbours classifier
    with feature subset combinations

    :param X: data frame of predictor features
    :param Y: column of labels
    :param domain: source domain from which the features are extracted ("TD" or "FD")
    :param k_neighbors: neighbours for k-nearest neighbours model to search in
    :param num_of_features: number of features in the subset to search in
    :param kfolds: number of splits for k-fold cross-validation
    :param power_transform: apply power transform of features in preprocessing instead
        of normalization
    :param model: name of the machine learning model to evaluate.
        Options are: "knn", "lda", "bayes", "svm"

    :return: training and testing accuracy for each
        hyperparameter and feature set combination
    """
    models = []
    for fnum in num_of_features:
        for k in k_neighbors:
            result = feature_combinations(
                X, Y, k, fnum, kfolds, domain, model,
                power_transform=power_transform
            )
            models.extend(result)

    return pd.DataFrame.from_records(models)


def accuracies_to_table(
            domain: str,
            set: str,
            distribution: pd.DataFrame,
            accuracy: pd.DataFrame
        ) -> dict:
    """Format accuracy to a row with metadata about hyperparamater and compute
    percentiles in the model accuracy distribution

    :param domain: source domain from which the features are extracted ("TD" or "FD")
    :param set: Title for the feature set or selection method
    :param distribution: accuracy distribution of the model
    :param accuracy: accuracy of the model in training and testing set

    :return: formatted structure for row of accuracies and percentiles
    """

    train_accuracy = accuracy['train']
    test_accuracy = accuracy['test']
    train_score = percentileofscore(distribution['train'].to_numpy(), train_accuracy)
    test_score = percentileofscore(distribution['test'].to_numpy(), test_accuracy)
    return {
        'domain': domain,
        'set': set,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_percentile': train_score,
        'test_percentile': test_score
    }


def feature_selection_accuracies(
            X: pd.DataFrame,
            Y: pd.DataFrame,
            domain: str,
            models_summary: pd.DataFrame,
            k_neighbors: int,
            number_of_features: int,
            power_transform: bool = False
        ) -> List[Dict[str, int]]:

    """Apply feature selection methods and evaluate accuracies for chosen number
    of neighbours and number of features

    :param X: data frame of predictor features
    :param Y: column of labels
    :param domain: source domain from which the features are extracted ("TD" or "FD")
    :params model_summary: accuracies from all feature subset combinations
    :param k_neighbors: neighbours for k-nearest neighbours model
    :param number_of_features: number of features in the subset
    :param power_transform: apply power transform of features in preprocessing instead
        of normalization

    :return: accuracies and percentiles for all tested feature selection methods
    """

    MODEL_TYPE = 'knn'
    kfolds = 5
    results = []

    accuracy_distribution = models_summary[
        (models_summary['domain'] == domain) &
        (models_summary['k'] == k_neighbors) &
        (models_summary['f'] == number_of_features)
    ].sort_values(by='train', ascending=False)

    y_best = all_features(
        X, Y, MODEL_TYPE,
        k_neighbors=[k_neighbors],
        power_transform=power_transform
    )
    y_best = {'train': y_best['train'][0], 'test': y_best['test'][0]}
    title = 'All features'
    r = accuracies_to_table(domain, title, accuracy_distribution, y_best)
    results.append(r)

    y_best = kfold_accuracy(
        transform_to_pca(X, n=number_of_features),
        Y, k_neighbors, kfolds, MODEL_TYPE,
        power_transform=power_transform
    )
    title = 'PCA PC'
    r = accuracies_to_table(domain, title, accuracy_distribution, y_best)
    results.append(r)

    y_best = (
        accuracy_distribution
        .head(1)
        .to_dict('records')[0]
    )
    title = 'Best features'
    r = accuracies_to_table(domain, title, accuracy_distribution, y_best)
    r['features'] = y_best['features']
    results.append(r)

    fsel_methods = [
        ('rank', 'Rank product'),
        ('corr', 'Correlation'),
        ('f_stat', 'F statistic'),
        ('mi', 'Mutual information')
    ]
    for pos, (name, title) in enumerate(fsel_methods):
        features = find_best_subset(X, Y, name, members=number_of_features)
        y_best = kfold_accuracy(
            X[list(features)], Y, k_neighbors, kfolds, MODEL_TYPE,
            power_transform=power_transform
        )
        r = accuracies_to_table(domain, title, accuracy_distribution, y_best)
        r['features'] = features
        results.append(r)

    return results


def model_boundaries(
            X: pd.DataFrame,
            Y: pd.DataFrame,
            n: int = 5,
            model_name: str = 'knn',
            knn_metric: str = 'euclidean'
        ):
    """Train k-nearest neighbours classifier to be used in determining
    its decision boundaries

    :param X: data frame of predictor features
    :param Y: column of labels
    :param n: number of neighbours for k-nearest neighbours model
    :param model_name: name of the machine learning model to evaluate.
        Options are: "knn", "lda", "bayes", "svm"
    :param knn_metric: distance metric name for k-nearest neighbours model

    :return: model fitted with training data of 80% from the original dataset
    """

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


def knn_online_learn(
            X: pd.DataFrame,
            Y: pd.DataFrame,
            window_len: int = 1,
            learn_skip: int = 0,
            clusters: int = False,
            n_neighbors: int = 5
        ) -> pd.DataFrame:

    """Progressive valuation of k-nearest neighbours classifier trained
    with increamental learning

    :param X: data frame of predictor features
    :param Y: column of labels
    :param window_len: Length of the tumbling window
    :param learn_skip: Gap of labeled observations in amount of samples
    :param clusters: return data points instead of valuation
    :param n_neighbors: number of neighbours for k-nearest neighbours model

    :return: performance of the model in progressive valuation over all generations
    """

    # Buffer true samples for learning for later: simulate delayed annotation
    learning_window = []

    # Model consists of scaler to give approximately same weight to all features and kNN
    scaler = preprocessing.MinMaxScaler()
    knn = neighbors.KNNClassifier(n_neighbors=n_neighbors)

    scores = []                 # List of tuples with accuracy, precision and recall score on each iteration
    v_true = []                 # Append y true sample on each iteration
    v_predict = []              # Append y predicted sample on each iteration

    skipping = 0
    started = False
    order_saved = []
    X['label'] = Y

    for idx, row in tqdm(X.iterrows()):
        x = {k: v for k, v in dict(row).items() if k != 'label'}
        x_scaled = scaler.learn_one(x).transform_one(x)
        y_true = row['label']
        learning_window.append((x_scaled, y_true))

        if started:
            # Predict sample after at least one example has been learned
            y_predict = knn.predict_one(x_scaled)
            v_true.append(y_true)
            v_predict.append(y_predict)
            order_saved.append(idx)

            scores.append([
                idx,
                skmetrics.accuracy_score(v_true, v_predict),
                skmetrics.precision_score(v_true, v_predict, average='micro'),
                skmetrics.recall_score(v_true, v_predict, average='micro')
            ])

        # Provide labels after window length has passed
        if len(learning_window) == window_len:
            for x, y in learning_window:
                # Learn first sample at start of window
                if skipping == learn_skip:
                    started = True
                    knn.learn_one(x, y)
                    skipping = 0
                else:
                    skipping += 1
            learning_window = []

    if clusters:
        return pd.Series(v_predict, index=order_saved)

    return pd.DataFrame(scores, columns=['step', 'accuracy', 'precision', 'recall'])
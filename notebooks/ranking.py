from enum import (
    Enum,
    auto
)
from typing import (
    List,
    Set,
    Tuple,
    Dict
)

import pandas as pd
from scipy.stats import gmean
from sklearn.feature_selection import (
    mutual_info_classif,
    f_classif
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from river import (
    feature_selection,
    stream,
    preprocessing
)
import selection


class ExperimentOutput(Enum):
    COUNTS = auto()
    BEST_SET = auto()
    RANKS = auto()
    SCORES_RANGE = auto()
    PCA = auto()
    SILHOUETTE = auto()
    BEST_CORR = auto()
    BEST_F_STAT = auto()
    BEST_MI = auto()


def silhouette_scores(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        Y_train: pd.DataFrame,
        Y_test: pd.DataFrame,
        best_features: list,
        pc: int) -> Dict[str, float]:
    Y_train = Y_train.reset_index(drop=True).astype('category')
    Y_test = Y_test.reset_index(drop=True).astype('category')

    scaler = MinMaxScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train)
    X_test[X_test.columns] = scaler.transform(X_test)    

    model = PCA(n_components=pc).fit(X_train)
    X_train_pca = pd.DataFrame(model.transform(X_train))
    X_test_pca = pd.DataFrame(model.transform(X_test))

    return {
        'train': silhouette_score(X_train[best_features], Y_train),
        'test': silhouette_score(X_test[best_features], Y_test),
        'train_pca': silhouette_score(X_train_pca, Y_train),
        'test_pca': silhouette_score(X_test_pca, Y_test)
    }


def pca_explained_variances(X_train: pd.DataFrame, pc: int) -> Dict[str, float]:
    scaler = MinMaxScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train)
    model = PCA(n_components=pc).fit(X_train)    
    return {f'PC{pc}': var for pc, var in enumerate(model.explained_variance_ratio_, start=1)}


def batch_feature_ranking(X: pd.DataFrame, Y: pd.DataFrame, mode: str = 'rank') -> pd.DataFrame:
    metric_ranks = pd.DataFrame()
    METRICS_OFFLINE = {
        'corr': selection.corr_classif, 
        'f_stat': f_classif,
        'mi': mutual_info_classif
    }

    if mode in METRICS_OFFLINE:
        metric = METRICS_OFFLINE[mode]
        scores = metric(X, Y)
        if isinstance(scores, tuple):
            scores = scores[0]
        leaderboard = (
            pd.DataFrame(zip(X.columns, scores), columns=['feature', 'rank'])
            .set_index('feature')
            .sort_values(by='rank', ascending=False)
        )
        return leaderboard

    elif mode == 'rank':
        for metric_name, metric in METRICS_OFFLINE.items():
            scores = metric(X, Y)
            if isinstance(scores, tuple):
                scores = scores[0]
            leaderboard = (
                pd.DataFrame(zip(X.columns, scores), columns=['feature', 'score'])
                .set_index('feature')
                .sort_values(by='score', ascending=False)
            )
            metric_ranks[metric_name] = leaderboard
        
        ranks = metric_ranks.rank(axis='rows', method='first', ascending=False)
        return ranks.apply(gmean, axis=1).sort_values().to_frame(name='rank')
    

def online_feature_ranking(X: pd.DataFrame, Y: pd.Series, mode: str = 'rank') -> pd.DataFrame:
    METRICS_ONLINE = {
        'corr': selection.Correlation, 
        'f_stat': selection.FisherScore,
        'mi': selection.MutualInformation
    }

    if mode in METRICS_ONLINE:
        metric = METRICS_ONLINE[mode]
        estimator = feature_selection.SelectKBest(similarity=metric(), k=2)
        for xs, ys in stream.iter_pandas(X, Y):
            estimator.learn_one(xs, ys)

        best = [dict(estimator.leaderboard.copy())]
        features = pd.DataFrame.from_records(best)

    elif mode == 'rank':
        estimators = [
            feature_selection.SelectKBest(similarity=metric(), k=2)
            for metric in METRICS_ONLINE.values()
        ]

        best = []
        for xs, ys in stream.iter_pandas(X, Y):
            for method in estimators:
                method.learn_one(xs, ys)

            scores = [method.leaderboard.copy() for method in estimators]
            scores = pd.DataFrame.from_records(scores).T
            ranks = scores.rank(axis='rows', method='first', ascending=False)
            ranks = ranks.apply(gmean, axis=1).to_dict()   # Smallest rank is the best
            best.append(ranks)   

        features = pd.DataFrame.from_records(best)

    return (
        features.tail(1)
        .reset_index(drop=True)
        .T.rename(columns={0: 'rank'})
        .reset_index()
        .rename(columns={'index': 'feature'})
        .set_index('feature')
        .sort_values(by='rank', ascending=True)
    )


def compute_correlations(X: pd.DataFrame, corr_above: float) -> Set[Tuple[str, str]]:
    corr = [
        {'feature_1': k[0], 'feature_2': k[1], 'corr': v}
        for k, v in X.corr().abs().stack().to_dict().items()
        if k[0] != k[1]
    ]
    corr = pd.DataFrame.from_records(corr)

    # Remove correlated features independent of tuple order
    correlations = corr[corr['corr'] >= corr_above][
        ['feature_1', 'feature_2']
    ].to_numpy()
    similar_pairs = set([(a, b) for a, b in correlations])
    similar_pairs.update([(b, a) for a, b in correlations])
    return  similar_pairs


def best_columns(ranks: pd.DataFrame, corr: Set[Tuple[str, str]], n: int) -> List[str]:
    columns = []
    for feature in ranks.index:
        # Make pairs with existing columns
        candidates = [
            col for col in columns 
            if (feature, col) in corr
        ]        
        # Append only if not correlation detected
        if len(candidates) == 0:
            columns.append(feature)

    # Limit to n features
    columns = columns[:n]
    return columns


def best_subset(ranks: pd.DataFrame, corr: Set[Tuple[str, str]], n: int) -> pd.DataFrame:
    columns = best_columns(ranks, corr, n)
    subset = ranks.copy()
    subset['rank'] = False
    subset[subset.index.isin(tuple(columns))] = True
    return subset
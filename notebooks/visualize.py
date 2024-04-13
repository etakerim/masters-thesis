import extraction
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

import seaborn as sb
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import models


def plot_models_performance_bar(
        X_temporal: pd.DataFrame,
        X_spectral: pd.DataFrame, 
        Y: pd.DataFrame,
        models_summary: pd.DataFrame,
        k_neighbors: int = 5,
        number_of_features: int = 3):

    MODEL_TYPE = 'knn'
    Y = Y.dropna().astype('category')
    X_temporal = X_temporal[X_temporal.index.isin(Y.index)].copy()
    X_spectral = X_spectral[X_spectral.index.isin(Y.index)].copy()
    Y = Y[Y.index.isin(X_temporal.index)].astype('category')

    X_temporal = X_temporal.reset_index(drop=True)
    X_spectral = X_spectral.reset_index(drop=True)
    Y = Y.reset_index(drop=True)
    Y = Y.cat.codes

    fig, ax = plt.subplots(1, 2, figsize=(10, 5)) 
    columns = ['train', 'test']
    kfolds = 5
    x = np.arange(len(columns))
    domains = {'temporal': X_temporal, 'spectral': X_spectral}

    for i, d in enumerate(domains.items()):
        domain, X = d 
        width = 0.13

        y_best = models.all_features(X, Y, MODEL_TYPE, [k_neighbors])
        y_best = [y_best['train'][0], y_best['test'][0]]
        rect = ax[i].bar(x - 3*width, y_best, width, label='All features')
        ax[i].bar_label(rect, padding=3, fmt=lambda x: f'{x * 100:.0f}')
        print('All features', y_best)

        y_best = kfold_accuracy(extraction.transform_to_pca(X, n=number_of_features), Y, k_neighbors, kfolds, MODEL_TYPE)
        y_best = [y_best['train'], y_best['test']]
        rect = ax[i].bar(x - 2*width, y_best, width, label='PCA 3 PC')
        ax[i].bar_label(rect, padding=3, fmt=lambda x: f'{x * 100:.0f}')
        print('PCA 3 PC', y_best)

        y_best = models_summary[
            (models_summary['domain'] == domain) &
            (models_summary['k'] == k_neighbors) & 
            (models_summary['f'] == number_of_features)
        ].sort_values(by='train', ascending=False).head(1).to_dict('records')[0]
        print('Best 3 features', y_best)
        y_best = [y_best['train'], y_best['test']]
        rect = ax[i].bar(x - 1*width, y_best, width, label='Best 3 features')
        ax[i].bar_label(rect, padding=3, fmt=lambda x: f'{x * 100:.0f}')
        print('Best 3 features', y_best)
   
        features = find_best_subset(X, Y, 'rank', number=number_of_features)
        y_best = kfold_accuracy(X[list(features)], Y, k_neighbors, kfolds, MODEL_TYPE)
        y_best = [y_best['train'], y_best['test']]
        rect = ax[i].bar(x - 0*width, y_best, width, label='Rank product')
        ax[i].bar_label(rect, padding=3, fmt=lambda x: f'{x * 100:.0f}')
        print('Rank product', y_best)

        features = find_best_subset(X, Y, 'corr', number=number_of_features)
        y_best = kfold_accuracy(X[list(features)], Y, k_neighbors, kfolds, MODEL_TYPE)
        y_best = [y_best['train'], y_best['test']]
        rect = ax[i].bar(x + 1*width, y_best, width, label='Correlation')
        ax[i].bar_label(rect, padding=3, fmt=lambda x: f'{x * 100:.0f}')
        print('Correlation', y_best)

        features = find_best_subset(X, Y, 'f_stat', number=number_of_features)
        y_best = kfold_accuracy(X[list(features)], Y, k_neighbors, kfolds, MODEL_TYPE)
        y_best = [y_best['train'], y_best['test']]
        rect = ax[i].bar(x + 2*width, y_best, width, label='F statistic')
        ax[i].bar_label(rect, padding=3, fmt=lambda x: f'{x * 100:.0f}')
        print('F statistic', y_best)
        
        features = find_best_subset(X, Y, 'mi', number=number_of_features)
        y_best = kfold_accuracy(X[list(features)], Y, k_neighbors, kfolds, MODEL_TYPE)
        y_best = [y_best['train'], y_best['test']]
        rect = ax[i].bar(x + 3*width, y_best, width, label='Mutual information')
        ax[i].bar_label(rect, padding=3, fmt=lambda x: f'{x * 100:.0f}')
        print('Mutual information', y_best)

        ax[i].set_xticks(x, columns)
        ax[i].legend(loc='lower right')
        ax[i].set_ylim(0.5, None)
        ax[i].set_title(domain)
        
        ax[i].set_ylabel('Accuracy')

    plt.tight_layout()
    plt.show()


# Scatter plot of best features with rank product
def scatter_features_3d(
        X: pd.DataFrame,
        Y: pd.DataFrame,
        features: list,
        size: tuple = (15, 5),
        boundary=False,
        model_name='knn'):

    Y = Y.dropna()
    X = X[X.index.isin(Y.index)].copy()
    Y = Y[Y.index.isin(X.index)].astype('category')

    scaler = MinMaxScaler()
    X[X.columns] = scaler.fit_transform(X)

    categories = Y.cat.categories
    colors = sb.color_palette('hls', len(categories))
    cmap = ListedColormap(colors.as_hex())

    fig, ax = plt.subplots(1, 3, figsize=size)

    for i, dims in enumerate([(0, 1), (0, 2), (1, 2)]):
        a, b = dims
        columns = [features[a], features[b]]
        p, q = X[columns[0]], X[columns[1]]
 
        if boundary:
            h = .02
            model = models.model_boundaries(X[columns], Y.cat.codes, model_name=model_name)
            x_min = p.min() - p.std()
            x_max = p.max() + p.std()
            y_min = q.min() - q.std()
            y_max = q.max() + q.std()
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

            Z = Z.reshape(xx.shape)
            ax[i].pcolormesh(xx, yy, Z, cmap=cmap, alpha=0.5)

        ax[i].scatter(
            p, q, c=Y.cat.codes, cmap=cmap, edgecolors='black'
        )

        legend_entries = []
        for c, n in dict(zip(Y.cat.codes, Y)).items():
            legend_entries.append(
                mpatches.Patch(color=colors[c], label=n)
            )
        ax[1].legend(handles=legend_entries)
        ax[i].set_xlabel(columns[0])
        ax[i].set_ylabel(columns[1])
        ax[i].grid(True)
    plt.show()


def scatter_features_3d_plot(
        X: pd.DataFrame,
        Y: pd.DataFrame,
        features: list,
        size: tuple = (8, 8),
        model_name='knn',
        boundary=False):
    Y = Y.dropna()
    X = X[X.index.isin(Y.index)].copy()
    Y = Y[Y.index.isin(X.index)].astype('category')

    X = X.reset_index(drop=True)
    Y = Y.reset_index(drop=True)

    scaler = MinMaxScaler()
    X_scaled = X.copy()
    X_scaled[X_scaled.columns] = scaler.fit_transform(X_scaled)

    categories = Y.astype('category').cat.categories
    colors = sb.color_palette('hls', len(categories))
    cmap = ListedColormap(colors.as_hex())

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(projection='3d')
    columns = [features[0], features[1], features[2]]

    model = models.model_boundaries(X_scaled[columns], Y, model_name=model_name)
    Y_predict = Y.to_frame('true')
    Y_predict['predict'] = pd.Series(model.predict(X_scaled[columns]))
    
    Y_good = Y_predict.loc[Y_predict['true'] == Y_predict['predict']]
    X_good = X[X.index.isin(Y_good.index)]
    xs, ys, zs = X_good[columns[0]], X_good[columns[1]], X_good[columns[2]]
    ax.scatter(xs, ys, zs, c=Y_good['true'].cat.codes, cmap=cmap, s=5)

    Y_bad = Y_predict.loc[Y_predict['true'] != Y_predict['predict']]
    X_bad = X[X.index.isin(Y_bad.index)]
    xs, ys, zs = X_bad[columns[0]], X_bad[columns[1]], X_bad[columns[2]]
    ax.scatter(xs, ys, zs, c=Y_bad['true'].cat.codes, marker='X', cmap=cmap, linewidths=1, edgecolors='black')

    legend_entries = []
    for c, n in dict(zip(Y.cat.codes, Y)).items():
        legend_entries.append(
            mpatches.Patch(color=colors[c], label=n)
        )
    ax.legend(handles=legend_entries)
    ax.set_xlabel(columns[0], labelpad=10)
    ax.set_ylabel(columns[1], labelpad=10)
    ax.set_zlabel(columns[2], labelpad=10)
    ax.grid(True)
    ax.view_init(elev=20, azim=-45)
    ax.set_box_aspect(None, zoom=0.85)
    fig.tight_layout()
    plt.show()


def project_classes(
        X: pd.DataFrame,
        Y: pd.DataFrame,
        size: tuple = (10, 8),
        boundary: bool = False,
        model_name: str = 'knn',
        pc: int = None):
    Y = Y.dropna()
    X = X[X.index.isin(Y.index)].copy()
    Y = Y[Y.index.isin(X.index)].astype('category')

    scaler = MinMaxScaler()
    X[X.columns] = scaler.fit_transform(X)

    print(silhouette_score(X, Y))

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca)

    categories = Y.cat.categories
    colors = sb.color_palette('hls', len(categories))
    cmap = ListedColormap(colors.as_hex())

    fig, ax = plt.subplots(1, 1, figsize=size)

    i, j = (0, 1) if pc is None else pc

    # KNN model
    if boundary:
        h = .02
        model = models.model_boundaries(X_pca, Y.cat.codes, model_name=model_name)
        x_min = X_pca[i].min() - X_pca[i].std()
        x_max = X_pca[i].max() + X_pca[i].std()
        y_min = X_pca[j].min() - X_pca[j].std()
        y_max = X_pca[j].max() + X_pca[j].std()
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.pcolormesh(xx, yy, Z, cmap=cmap, alpha=0.5)


    ax.scatter(X_pca[0], X_pca[1], c=Y.cat.codes, cmap=cmap, edgecolors='black')

    legend_entries = []
    for c, n in dict(zip(Y.cat.codes, Y)).items():
        if n != 'nan':
            legend_entries.append(
                mpatches.Patch(color=colors[c], label=n)
            )

    ax.legend(handles=legend_entries)

    var = 100 * pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var[0]:.2f} %)')
    ax.set_ylabel(f'PC2 ({var[1]:.2f} %)')
    ax.grid(True)
    plt.show()


def project_classes_3d(X: pd.DataFrame, Y: pd.DataFrame, size=(15, 4)):
    Y = Y.dropna()
    X = X[X.index.isin(Y.index)].copy()
    Y = Y[Y.index.isin(X.index)].astype('category')

    scaler = MinMaxScaler()
    X[X.columns] = scaler.fit_transform(X)

    print(silhouette_score(X, Y))

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca)

    categories = Y.cat.categories
    colors = sb.color_palette('hls', len(categories))
    fig, ax = plt.subplots(1, 3, figsize=size)

    for pos, dim in enumerate([(0, 1), (0, 2), (1, 2)]):
        i, j = dim
        for label, color in zip(categories, colors):
            rows = list(Y[Y == label].index)
            x = X_pca[X_pca.index.isin(rows)][i]
            y = X_pca[X_pca.index.isin(rows)][j]
            ax[pos].scatter(x, y, s=2, color=color, label=label)

        var = 100 * pca.explained_variance_ratio_
        ax[pos].set_xlabel(f'PC{i+1} ({var[i]:.2f} %)')
        ax[pos].set_ylabel(f'PC{j+1} ({var[j]:.2f} %)')
        ax[pos].grid(True)
        ax[pos].legend()

    fig.tight_layout()
    plt.show()


def cross_cuts_3d(X_train, y_train, ylim=None):
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    labels = np.unique(y_train.to_numpy())
    colors = sb.color_palette('hls', len(labels))

    for i, axes in enumerate(((0, 1), (0, 2), (1, 2))):
        a, b = axes

        for label, color in zip(labels, colors):
            x = X_train.loc[
                list(y_train[y_train == label].index), 
                X_train.columns[a]
            ]
            y = X_train.loc[
                list(y_train[y_train == label].index),
                X_train.columns[b]
            ]
            ax[i].scatter(x, y, s=1, color=color, label=label)
            if ylim is not None:
                ax[i].set_ylim(ylim)
            
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
        x = df.loc[:, df.columns[a]]
        y = df.loc[:, df.columns[b]]
        ax[i].scatter(x, y, color='grey', s=1)

        for flag, color in ((False, 'green'), (True, 'red')):
            points = list(df[df['anomaly'] == flag].index)
            x = df.loc[points, df.columns[a]]
            y = df.loc[points, df.columns[b]]
            ax[i].scatter(x, y, color=color, s=1)
    
        ax[i].set_xlabel(df.columns[a])
        ax[i].set_ylabel(df.columns[b])
        ax[i].grid()
        ax[i].legend()


def scatter_classif(X, y_label, categories, colors, ax):
    for label, color in zip(categories, colors):
            rows = list(y_label[y_label == label].index)
            x = X.loc[rows, 0]
            y = X.loc[rows, 1]
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
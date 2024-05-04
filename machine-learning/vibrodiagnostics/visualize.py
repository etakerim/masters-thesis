from typing import List
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from adjustText import adjust_text

import seaborn as sb
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from scipy.signal import freqz

from vibrodiagnostics import extraction, models


DOMAIN_TITLES = {
    'TD': 'Time domain',
    'FD': 'Frequency domain',
    'TD+FD': 'Time and Frequency domain'
}



def evolution_of_severity_levels(df: pd.DataFrame):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(np.arange(0, len(df)), df['severity_level'], color='red')
    ax.set_xlabel('Observations')
    ax.set_ylabel('Severity level')
    ax.grid()
    plt.show()


def plot_models_performance_bar(results: pd.DataFrame) -> pd.DataFrame:
    if len(results['domain'].unique()) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax = [ax]
    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    width = 0.13
    columns = ['train', 'test']
    x = np.arange(len(columns))

    for i, (domain_name, domain) in enumerate(results.groupby(by=['domain'])):
        for pos, (name, scenario) in enumerate(domain.iterrows(), start=-3):
            title = scenario['set']
            y = scenario['train_accuracy'], scenario['test_accuracy']
            rect = ax[i].bar(x + pos*width, y, width, label=title)

            ax[i].bar_label(rect, padding=3, fmt=lambda x: f'{x * 100:.0f}')
            ax[i].set_xticks(x, columns)
            ax[i].legend(loc='lower right')
            ax[i].set_ylim(0.5, None)
            ax[i].set_title(DOMAIN_TITLES[domain_name[0]])
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
        model_name='knn',
        power_transform: bool = False):

    if power_transform is True:
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    else:
        scaler = MinMaxScaler()

    X = X.copy()
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
            table = np.c_[xx.ravel(), yy.ravel()]

            table = pd.DataFrame(table)
            table.columns = list(columns)
            Z = model.predict(table)
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
    fig.tight_layout()
    plt.show()


def scatter_features_3d_plot(
        X: pd.DataFrame,
        Y: pd.DataFrame,
        features: list,
        size: tuple = (8, 8),
        model_name: str ='knn',
        boundary: bool = False,
        power_transform: bool = False):

    if power_transform is True:
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    else:
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


def project_classes_3d(X: pd.DataFrame, Y: pd.DataFrame, size=(15, 6)):
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


def cross_cuts_3d(X_train: pd.DataFrame, y_train: pd.DataFrame, ylim=None):
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


def cross_cuts_3d_cluster(X_train, cluster, title):
    df = X_train.copy()
    df['cluster'] = cluster
    df['cluster'] = df['cluster'].astype('category')

    categories = df['cluster'].cat.categories
    colors = sb.color_palette('hls', len(categories))
    fig, ax = plt.subplots(1, 3, figsize=(15, 3))
    fig.suptitle(title)

    for i, axes in enumerate(((0, 1), (0, 2), (1, 2))):
        a, b = axes
         
        for label, color in zip(categories, colors):
            rows = list(df[df['cluster'] == label].index)
            x = df.loc[rows, df.columns[a]]
            y = df.loc[rows, df.columns[b]]
            ax[i].scatter(x, y, s=1, color=color, label=label)

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


def plot_cumulative_explained_variance(td_variance: np.array, fd_variance: np.array):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        np.arange(1, len(td_variance) + 1),
        100 * np.cumsum(td_variance), 
        marker='s', label='Features in time domain'
    )
    ax.plot(
        np.arange(1, len(fd_variance) + 1), 
        100 * np.cumsum(fd_variance),
        marker='s', label='Features in frequency domain'
    )
    ax.set_xlabel('Number of principal components')
    ax.set_ylabel('Explained variance [%]')
    ax.grid()
    ax.legend()
    plt.show()


def loading_plot(loadings: list, feature_names: List[str], bottom: float, top: float):
    xs = loadings[0]
    ys = loadings[1]

    texts = []
    # Plot the loadings on a scatterplot
    for i, varnames in enumerate(feature_names):
        plt.arrow(
            0, 0,   # coordinates of arrow base
            xs[i],  # length of the arrow along x
            ys[i],  # length of the arrow along y
            color='r', 
            head_width=0.01
        )
        texts.append(plt.text(xs[i], ys[i], varnames))

    # Define the axis
    adjust_text(texts, only_move={'points':'y', 'texts':'y'})
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim(bottom, top)
    plt.ylim(bottom, top)
    plt.grid()
    plt.show()
    


def plot_all_knn(td_results: dict, fd_results: dict):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(td_results['k'], td_results['train'], marker='x', color='darkblue', label='train - temporal')
    ax.plot(td_results['k'], td_results['test'], marker='x', color='blue', label='test - temporal')

    ax.plot(fd_results['k'], fd_results['train'], marker='x', color='darkgreen', label='train - spectral')
    ax.plot(fd_results['k'], fd_results['test'], marker='x', color='green', label='test - spectral')

    ax.set_ylabel(f'Accuracy')
    ax.set_xlabel('K-neighbors')
    ax.set_xticks(td_results['k'])
    ax.grid(True)
    ax.legend()
    plt.show()


def plot_all_knn_simple(results: dict):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(results['k'], results['train'], marker='x', color='darkblue', label='training set')
    ax.plot(results['k'], results['test'], marker='x', color='blue', label='testing set')

    ax.set_ylabel(f'Accuracy')
    ax.set_xlabel('K-neighbors')
    ax.set_xticks(results['k'])
    ax.grid(True)
    ax.legend()
    plt.show()


def boxplot_enumerate_models_accuracy(results: pd.DataFrame, metric, plots_col: str, inplot_col: str):
    for fnum, features in results.groupby(by=plots_col):
        if len(features['domain'].unique()) == 1:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            ax = [ax]
        else:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    
        for i, group in enumerate(features.groupby(by='domain', sort=False)):
            domain_name, domain = group 
            ax[i].grid()
            domain_name = DOMAIN_TITLES[domain_name]
            
            if plots_col == 'k':
                ax[i].set_title(f'K-neighbors: {fnum}, {domain_name}')
            if plots_col == 'f':
                ax[i].set_title(f'Features: {fnum}, {domain_name}')

            boxplot_data = {}
            for k, models in domain.groupby(by=[inplot_col]):
                print(fnum, k, models[metric].describe())
                boxplot_data[k[0]] = models[metric].to_list()

            ax[i].boxplot(
                boxplot_data.values(),
                labels=boxplot_data.keys(),
                medianprops={'linewidth': 2, 'color': 'black'},
                notch=True
            )
            ax[i].set_ylabel('Accuracy')
            if plots_col == 'f':
                ax[i].set_xlabel('K-neighbors')
            if plots_col == 'k':
                ax[i].set_xlabel('Number of features')
    plt.show()


def plot_label_occurences(y):
    observations = []
    columns = list(y.astype('category').cat.categories)
    empty = dict(zip(columns, len(columns) * [0]))

    for row in y.astype('category'):
        sample = empty.copy()
        sample[row] = 1
        observations.append(sample)

    class_occurences = pd.DataFrame.from_records(observations).cumsum()
    ax = class_occurences.plot(grid=True, figsize=(10, 5), xlabel='Observations', ylabel='Label occurences')
    return ax, class_occurences


def plot_filter_response(b, a, title=''):
    w, h = freqz(b, a, fs=50000)
    fig, ax = plt.subplots()
    ax.plot(w, 20 * np.log10(abs(h)), 'b')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [dB]')
    ax.set_title(title)
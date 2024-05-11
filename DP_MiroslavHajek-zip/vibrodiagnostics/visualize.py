from typing import List, Dict
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
"""
Titles for signal source domain abbreviation
"""


def evolution_of_severity_levels(df: pd.DataFrame):
    """Line chart of the amount of observations at relative severity levels

    :param df: data frame with sorted "severity" level column 
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(np.arange(0, len(df)), df['severity_level'], color='red')
    ax.set_xlabel('Observations')
    ax.set_ylabel('Severity level')
    ax.grid()
    plt.show()


def plot_models_performance_bar(results: pd.DataFrame):
    """Bar chart of feature selection accuracy comparison

    :param results: training and testing accuracies of seven feature
        selection methods in both source domains
    """
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
        boundary: bool = False,
        model_name: str = 'knn',
        power_transform: bool = False):
    """Scatter plot of data points in 3D feature space shown 
    in planar cross-sections through coordinate axes 

    :param X: data frame of features
    :param Y: labels of observations
    :param features: names of three features to display
    :param size: figure size
    :param boundary: show decision boundary for k-NN model with 5 neighbours
    :param model_name: name of the machine learning model to evaluate. 
        Options are: "knn", "lda", "bayes", "svm"
    :param power_transform: apply power transform of features in preprocessing instead 
        of normalization
    """

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
        boundary: bool = False,
        model_name: str ='knn',
        power_transform: bool = False):
    """Three dimensional scatter plot of data points in feature space

    :param X: data frame of features
    :param Y: labels of observations
    :param features: names of three features to display
    :param size: figure size
    :param boundary: show decision boundary for k-NN model with 5 neighbours
    :param model_name: name of the machine learning model to evaluate. 
        Options are: "knn", "lda", "bayes", "svm"
    :param power_transform: apply power transform of features in preprocessing instead 
        of normalization
    """

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
    """Scatter plot of two principal components of data points in feature space

    :param X: data frame of features
    :param Y: labels of observations
    :param size: figure size
    :param boundary: show decision boundary for k-NN model with 5 neighbours
    :param model_name: name of the machine learning model to evaluate. 
        Options are: "knn", "lda", "bayes", "svm"
    :param pc: number of principal components
    """

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


def project_classes_3d(X: pd.DataFrame, Y: pd.DataFrame, size: tuple = (15, 6)):
    """Scatter plot of three principal components of data points in feature space
    shown in planar cross-sections through coordinate axes 

    :param X: data frame of features
    :param Y: labels of observations
    :param size: figure size
    """
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


def cross_cuts_3d_cluster(X_train: pd.DataFrame, cluster: str, title: str):
    """Scatter plot of clusters in 3D feature space shown 
    in planar cross-sections through coordinate axes 

    :param X_train: data frame of features
    :param cluster: clusters that observations belong to
    :param title: figure title
    """
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


def scatter_classif(
            X: pd.DataFrame,
            y_label: pd.Series,
            categories: List[str],
            colors: List[str],
            ax):
    """Scatter plot of data points with color based on their labels

    :param X: data frame of features
    :param y_label: labels of observations
    :param categories: list of unique clasess
    :param colors: list of colors for clasess
    :param ax: subplot axis
    """
    for label, color in zip(categories, colors):
            rows = list(y_label[y_label == label].index)
            x = X.loc[rows, 0]
            y = X.loc[rows, 1]
            ax.scatter(x, y, s=2, color=color, label=label)


def project_classifier_map_plot(X: pd.DataFrame, y_true: pd.Series, y_predict: pd.Series):
    """Scatter plots of two prinicipal components from data points that shows mistakes
    in prediction versus true labels

    :param X: data frame of features
    :param y_true: true labels of observations
    :param y_predict: predicted labels of observations
    """
    y_true = y_true.astype('category') 
    y_predict = y_predict.astype('category')

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca)

    categories = y_true.cat.categories
    colors = sb.color_palette('hls', len(categories))

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


def plot_cumulative_explained_variance(td_variance: np.array, fd_variance: np.array):
    """Line chart of relationship of number of principal components 
    to total explained variance

    :param td_variance: Explained variances for time-domain features
    :param fd_variance: Explained variances for frequency-domain features
    """
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
    """Loading plot of features

    :param loadings: Relation of features to coordinates that are 
        created by two principal components
    :param feature_names: list of feature names corresponding to their loadings
    :param bottom: lower limit of graph coordinates in x and y axes
    :param top: upper limit of graph coordinates in x and y axes
    """
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


def plot_all_knn(td_results: Dict[str, float], fd_results: Dict[str, float]):
    """Line chart of relationship of k-value to k-NN classifier accuracy 

    :param td_results: lists of k-values and accuracies for time-domain features
        in training and testing set
    :param fd_results: lists of k-values and accuracies for frquency-domain features
        in training and testing set
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(td_results['k'], td_results['train'], marker='x', color='darkblue', label='train - temporal')
    ax.plot(td_results['k'], td_results['test'], marker='x', color='blue', label='test - temporal')

    ax.plot(fd_results['k'], fd_results['train'], marker='x', color='darkgreen', label='train - spectral')
    ax.plot(fd_results['k'], fd_results['test'], marker='x', color='green', label='test - spectral')

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('K-neighbors')
    ax.set_xticks(td_results['k'])
    ax.grid(True)
    ax.legend()
    plt.show()


def boxplot_enumerate_models_accuracy(
        results: pd.DataFrame,
        metric: str,
        plots_col: str,
        inplot_col: str):
    """Boxplot of model accuracy distributions for various 
        number of features or neighbours

    :param results: model accuracy distributions
    :param metric: column of values to show in values for accuracy
    :param plots_col: constant parameter for subplot ("f" or "k")
    :param inplot_col: comparison of different values for the parameter 
        within subplot ("f" or "k")
    """
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
                # ax[i].set_title(f'K-neighbors: {fnum}, {domain_name}')
                print(f'K-neighbors: {fnum}, {domain_name}')
            if plots_col == 'f':
                # ax[i].set_title(f'Features: {fnum}, {domain_name}')
                print(f'Features: {fnum}, {domain_name}')

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
            ax[i].set_ylim(0.5, 1)
            ax[i].set_ylabel('Accuracy')
            if plots_col == 'f':
                ax[i].set_xlabel('K-neighbors')
            if plots_col == 'k':
                ax[i].set_xlabel('Number of features')
    plt.show()


def plot_label_occurences(y: pd.Series):
    """Line chart of counters for classes in incremental learning

    :param y: sorted labels of observations
    """
    observations = []
    columns = list(y.astype('category').cat.categories)
    empty = dict(zip(columns, len(columns) * [0]))

    for row in y.astype('category'):
        sample = empty.copy()
        sample[row] = 1
        observations.append(sample)

    class_occurences = pd.DataFrame.from_records(observations).cumsum()
    ax = class_occurences.plot(
        grid=True,
        figsize=(10, 5),
        xlabel='Observations',
        ylabel='Label occurences'
    )
    return ax, class_occurences
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import itertools


def knn_evaluation(x_train, y_train, x_test, y_test, n=5):
    knn = KNeighborsClassifier(n_neighbors=n, metric='euclidean', algorithm='kd_tree')
    knn.fit(x_train, y_train)
    y_predict_train = knn.predict(x_train)
    y_predict_test = knn.predict(x_test)

    print(f'Train accuracy: {metrics.accuracy_score(y_train, y_predict_train) * 100:.2f} %')
    print(f'Test accuracy: {metrics.accuracy_score(y_test, y_predict_test) * 100:.2f} %')
    print(metrics.classification_report(y_test, y_predict_test))

    labels = np.unique(y_test)
    cm = metrics.confusion_matrix(y_test, y_predict_test)
    cm = pd.DataFrame(cm, index=labels, columns=labels)

    ax = sb.heatmap(cm, cbar=True, cmap='BuGn', annot=True, fmt='d')
    ax.set_xlabel('Predicted label')    # size=15)
    ax.set_ylabel('True label')         # size=15)
    plt.show()


def knn_one_case_eval(neighbours, features, x_train, y_train, x_test, y_test):
    x_train_selected = x_train[features]
    x_test_selected = x_test[features]

    knn = KNeighborsClassifier(n_neighbors=neighbours, metric='euclidean', algorithm='kd_tree')
    knn.fit(x_train_selected, y_train)
    y_predict_train = knn.predict(x_train_selected)
    y_predict_test = knn.predict(x_test_selected)

    y_proba_train = knn.predict_proba(x_train_selected)
    y_proba_test = knn.predict_proba(x_test_selected)

    return {
        'features': features,
        'train_accuracy': metrics.accuracy_score(y_train, y_predict_train),
        'train_precision': metrics.precision_score(y_train, y_predict_train, average='micro'),
        'train_recall': metrics.recall_score(y_train, y_predict_train, average='micro'),
        'train_error_rate': np.mean(y_train != y_predict_train),
        # 'train_auc': metrics.roc_auc_score(y_train, y_proba_train, multi_class='ovo', average='macro'),
        'test_accuracy': metrics.accuracy_score(y_test, y_predict_test),
        'test_precision': metrics.precision_score(y_test, y_predict_test, average='micro'),
        'test_recall': metrics.recall_score(y_test, y_predict_test, average='micro'),
        'test_error_rate': np.mean(y_test != y_predict_test)
        #'test_auc': metrics.roc_auc_score(y_test, y_proba_test, multi_class='ovo', average='macro')
    }


def knn_feature_combinations(neighbours, all_features, combinations, x_train, y_train, x_test, y_test):
    evaluation = []

    for features in tqdm(itertools.combinations(all_features, r=combinations)):
        result = knn_one_case_eval(neighbours, list(features), x_train, y_train, x_test, y_test)
        evaluation.append(result)

    evaluation = pd.DataFrame.from_records(evaluation)
    return evaluation.sort_values(by='train_accuracy', ascending=False).reset_index(drop=True)

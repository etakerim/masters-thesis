import numpy as np
import pandas as pd
from river import (
    stats,
    preprocessing
)
from scipy.stats import pointbiserialr


class Correlation(stats.base.Bivariate):

    def __init__(self):
        self.labels = {}

    def update(self, x, y):
        if y not in self.labels:
            self.labels[y] = stats.PearsonCorr()

        for label, corr in self.labels.items():
            if label == y:
                corr.update(x, 1)
            else:
                corr.update(x, 0)  
        return self

    def get(self):
        return np.mean([
            abs(corr.get())
            for corr in self.labels.values()
        ])


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
                    score += p_xy * np.log(p_xy / (p_x * p_y))

        return score


def corr_classif(X, y):
    X = pd.DataFrame(X)
    y_dummies = pd.get_dummies(y)
    scores = []

    for col in X.columns:
        x = X[col]

        if (x == x.iloc[0]).all():
            # Correlation of constant list is not defined
            # Say that it is not infomative by corr = 0
            scores.append(0)   
        else:
            corr = np.array([
                np.abs(pointbiserialr(y_dummies[category], x)[0])
                for category in np.unique(y)
            ])
            scores.append(corr.mean())

    return np.array(scores)
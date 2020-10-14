from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import math
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def load_train_test_data(test_ratio=.3, random_state=1):
    balance_scale = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data",
                                names=['Class Name', 'Left-Weigh', 'Left-Distance', 'Right-Weigh', 'Right-Distance'], header=None)
    # print('balance_scale',balance_scale)
    class_le = LabelEncoder()
    balance_scale['Class Name'] = class_le.fit_transform(
        balance_scale['Class Name'].values)
    # print('balance_scale',balance_scale['Class Name'])
    X = balance_scale.iloc[:, 1:].values
    # X = 四個特徵的值 Left-Weigh', 'Left-Distance', 'Right-Weigh','Right-Distance
    # print('balance_scale_x',X)
    y = balance_scale['Class Name'].values
    # print('balance_scale_y',y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std


def main():
    X_train, X_test, y_train, y_test = load_train_test_data(
        test_ratio=.3, random_state=1)
    X_train_scale, X_test_scale = scale_features(X_train, X_test)

    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X_train_scale, y_train)
    y = clf.predict(X_test)
    print('accuracy_score', accuracy_score(y_test, y))

    plt.figure(figsize=(25, 10))
    plottree(clf)
    print(plot_tree(clf))


def plottree(clf):
    plot_tree(clf)


if __name__ == '__main__':
    X, y = datasets.load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1024)

    main()

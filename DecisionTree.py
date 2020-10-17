from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics
import numpy as np
import math
import pandas as pd


class Node:
    """A decision tree node."""

    def __init__(self, gini, entropy, num_samples,
                 num_samples_per_class, predicted_class):
        self.gini = gini
        self.entropy = entropy
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        
        self.leaf = bool
        self.left_X_train_scale = []
        self.right_X_train_scale = []
        self.left_y_train = []
        self.right_y_train = []


class DecisionTreeClassifier:
    def __init__(self, criterion='gini', max_depth=4):
        self.criterion = criterion
        self.max_depth = max_depth

    def _gini(self, sample_y, n_classes):
        total_num_sample = sample_y.size
        elements, counts = np.unique(sample_y, return_counts=True)
        gini = 0
        for i in counts:
            gini = gini + (i/total_num_sample)*(1-i/total_num_sample)
        return gini

    def _entropy(self, sample_y, n_classes):
        elements, counts = np.unique(sample_y, return_counts=True)
        entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i] /
                                                              np.sum(counts)) for i in range(len(elements))])
        return entropy
    # X_train_scale, y_train
    def _feature_split(self, X, y, n_classes, node):
        m = y.size
        if m <= 1:
            return None, None
        
        # Gini or Entropy of current node.
        if self.criterion == "gini":
            best_criterion = self._gini(y, n_classes)
        else:
            best_criterion = self._entropy(y, n_classes)

        best_idx, best_thr = None, None

        information = []
        for j in range(4):
            tem_train_scale = X
            index = np.argsort(tem_train_scale[:, j])
            sort_x = tem_train_scale[:, j][index]
            sort_y = y[index]

            find_midpoint = []
            for i in range(len(index)-1):
                if(sort_x[i] < sort_x[i+1]):
                    midpoint = (sort_x[i]+sort_x[i+1])/2
                    find_midpoint.append([round(midpoint, 3), i+1])
            
            if(len(find_midpoint) == 0):
                continue
            for i in range(len(find_midpoint)):
                left = sort_y[:find_midpoint[i][1]]
                right = sort_y[find_midpoint[i][1]:]
                if self.criterion == "gini":
                    left_criterion_value = round(
                        self._gini(left, n_classes), 3)
                    right_criterion_value = round(
                        self._gini(right, n_classes), 3)
                    information.append(
                        [left_criterion_value, right_criterion_value, find_midpoint[i][1], j,find_midpoint[i][0]])
                else:
                    left_criterion_value = round(
                        self._entropy(left, n_classes), 3)
                    right_criterion_value = round(
                        self._entropy(right, n_classes), 3)
                    information.append(
                        [left_criterion_value, right_criterion_value, find_midpoint[i][1], j,find_midpoint[i][0]])
        # 找最小的entropy or gini index
        return_information = []
        data = []
        sample = m
        for i in range(len(information)):
            left_sample = information[i][2]
            right_sample = sample - left_sample
            left_entropy = information[i][0]
            right_entropy = information[i][1]
            result = (left_sample/sample)*left_entropy + (right_sample/sample)*right_entropy
            data.append(result)
    
        for i in range(len(information)):
            if min(data) == data[i]:
                for j in range(len(information[i])):
                    return_information.append(information[i][j])
        
        best_idx, best_thr = return_information[3], return_information[4]

        index_result = np.argsort(tem_train_scale[:, return_information[3]])
        sort_x_result = tem_train_scale[index_result]
        sort_y_result = y[index_result]
 
        # 返回左右子樹的X_train_scale && y_train data
        node.left_X_train_scale = sort_x_result[:return_information[2]]
        node.right_X_train_scale = sort_x_result[return_information[2]:]
        node.left_y_train = sort_y_result[:return_information[2]]
        node.right_y_train = sort_y_result[return_information[2]:]

        if best_thr == 0 or (return_information[0] ==0 and return_information[1] ==0):
            print('This node is leaf')
            print('return pre node')
            node.leaf = True
            return None, None
        else:
            print('find idx is : {0}, thr is : {1}'.format(best_idx, best_thr))
            print('Left node ' + self.criterion + ' is : '+ str(return_information[0]))
            print('Right node ' + self.criterion + ' is : '+ str(return_information[1]))
        return best_idx, best_thr

    # X_train_scale, y_train
    def _build_tree(self, X, y, depth=2):
        
        print('bulid new node')
        num_samples_per_class = [np.sum(y == i)for i in range(self.n_classes_)]
        print('num_samples_per_class', num_samples_per_class)
        # 暫且不明白要做啥
        predicted_class = np.argmax(num_samples_per_class)

        node = Node(
            gini=self._gini(y, self.n_classes_),
            entropy=self._entropy(y, self.n_classes_),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class, 
        )

        if depth < self.max_depth:
            node.leaf = False
            self.n_classes_ = len(np.unique(y))
            idx, thr = self._feature_split(
                X, y, self.n_classes_, node)
            node.feature_index = idx
            node.threshold = thr

            if idx is not None:
                print('\n')
                print('bulid left tree <-----')
                node.left = self._build_tree(
                    node.left_X_train_scale, node.left_y_train, depth+1)
                print('\n')
                print('bulid right tree ----->')
                node.right = self._build_tree(
                    node.right_X_train_scale, node.right_y_train, depth+1)
        else:
            print('depth >= max_depth')
        
        return node
    
    # X_train_scale, y_train
    def fit(self, X, Y):
        self.n_classes_ = len(np.unique(Y))
        self.n_features_ = X.shape[1]
        # if user entered a value which was neither gini nor entropy
        if self.criterion != 'gini':
            if self.criterion != 'entropy':
                self.criterion = 'gini'
        self.tree_ = self._build_tree(X, Y)
    # X =  X_test_scale
    def predict(self, X):
        predicted_class = self.predict_class_fun(X)
        pred = predicted_class[:,0]
        
        return pred
    def predict_class_fun(self,X):
        predicted_class = np.empty((X.shape[0],self.n_classes_))
        for i in range(X.shape[0]):
            predicted_class[i] = self.predict_row(X[i,:],self.tree_)
        return predicted_class
    def predict_row(self,row,tree_):
        """Predict single row"""
        if tree_.leaf:
            return tree_.predicted_class
        else:
            if row[tree_.feature_index]<=tree_.threshold:
                return self.predict_row(row,tree_.left)
            else:
                return self.predict_row(row,tree_.right)
            

def load_train_test_data(test_ratio=.3, random_state=1):
    balance_scale = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data",
                                names=['Class Name', 'Left-Weigh', 'Left-Distance', 'Right-Weigh', 'Right-Distance'], header=None)
    class_le = LabelEncoder()
    balance_scale['Class Name'] = class_le.fit_transform(
        balance_scale['Class Name'].values)
    X = balance_scale.iloc[:, 1:].values
    y = balance_scale['Class Name'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std


def accuracy_report(X_train_scale, y_train, X_test_scale, y_test, criterion='gini', max_depth=4):
    tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    tree.fit(X_train_scale, y_train)
    
    print('\n---start predict----\n')
    pred = tree.predict(X_train_scale)
    
    print(criterion + " tree train accuracy: %f"
          % (sklearn.metrics.accuracy_score(y_train, pred)))
    pred = tree.predict(X_test_scale)
    print(criterion + " tree test accuracy: %f\n"
          % (sklearn.metrics.accuracy_score(y_test, pred)))


def main():
    X_train, X_test, y_train, y_test = load_train_test_data(
        test_ratio=.3, random_state=1)
    X_train_scale, X_test_scale = scale_features(X_train, X_test)
    
    print('\n--------start--------\n')
    # gini tree
    print('Bulid Decision Tree by gini')
    accuracy_report(X_train_scale, y_train, X_test_scale,
                    y_test, criterion='gini', max_depth=6)
    # entropy tree
    print('Bulid Decision Tree by entropy')
    accuracy_report(X_train_scale, y_train, X_test_scale,
                    y_test, criterion='entropy', max_depth=6)
    
    print('\n--------finish--------\n')


if __name__ == "__main__":
    main()

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

class DecisionTreeClassifier:
    def __init__(self,criterion='gini', max_depth=4):
        self.criterion = criterion
        self.max_depth = max_depth

    def _gini(self,sample_y,n_classes):
        total_num_sample = sample_y.size
        elements,counts = np.unique(sample_y,return_counts = True)
        print('elements,counts ',elements,counts)
        gini = 0
        for i in counts:
            gini = gini + (i/total_num_sample)*(1-i/total_num_sample)
        print('gini number:',gini)
        return gini
         
    def _entropy(self,sample_y,n_classes):
        elements,counts = np.unique(sample_y,return_counts = True)
        print('elements,counts ',elements,counts )
        entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
        return entropy
    #X_train_scale, y_train 參考投影片 page.28
    def _feature_split(self, X, y,n_classes):
        # Returns:
        #  best_idx: Index of the feature for best split, or None if no split is found.
        #  best_thr: Threshold to use for the split, or None if no split is found.
        
        # print(X.shape(1))

        # for i in range(n_classes+1):
        #     self._gini(X[:,i],n_classes)

        self._gini(y,n_classes)
        elements,counts = np.unique(y,return_counts = True)
        print('2_feature_split elements,counts ',elements,counts )
        m = y.size
        if m <= 1:
            return None, None

        # Gini or Entropy of current node.
        if self.criterion == "gini":
            best_criterion = self._gini(y,n_classes)
        else:
            best_criterion = self._entropy(y,n_classes)

        best_idx, best_thr = None, None

        # TODO: find the best split, loop through all the features, and consider all the
        # midpoints between adjacent training samples as possible thresholds. 
        # Computethe Gini or Entropy impurity of the split generated by that particular feature/threshold
        # pair, and return the pair with smallest impurity.

        return best_idx, best_thr

    def _build_tree(self, X, y, depth=0):
        
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y,self.n_classes_),
            entropy = self._entropy(y,self.n_classes_),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )
        
        if depth < self.max_depth:
            idx, thr = self._feature_split(X, y,self.n_classes_)
            if idx is not None:
            # TODO: Split the tree recursively according index and threshold until maximum depth is reached.
                pass

        return node
    #X_train_scale, y_train
    def fit(self,X,Y):
        # Fits to the given training data
        self.n_classes_ = len(np.unique(Y)) 
        # n_features_ 有四個特徵 len = 437 
        self.n_features_ = X.shape[1]
        
        # if user entered a value which was neither gini nor entropy
        if self.criterion != 'gini' :
            if self.criterion != 'entropy':
                self.criterion='gini'         
        self.tree_ = self._build_tree(X, Y)
    # X =  X_test_scale
    def predict(self,X):
        pred = []
        #TODO: predict the label of data
        return pred

def load_train_test_data(test_ratio=.3, random_state = 1):
    balance_scale = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data",
            names=['Class Name', 'Left-Weigh', 'Left-Distance', 'Right-Weigh','Right-Distance'],header=None)
    # print('balance_scale',balance_scale)
    class_le = LabelEncoder()
    balance_scale['Class Name'] = class_le.fit_transform(balance_scale['Class Name'].values)
    # print('balance_scale',balance_scale['Class Name'])
    X = balance_scale.iloc[:,1:].values
    # X = 四個特徵的值 Left-Weigh', 'Left-Distance', 'Right-Weigh','Right-Distance
    # print('balance_scale_x',X)
    y = balance_scale['Class Name'].values
    # print('balance_scale_y',y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_ratio, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std , X_test_std

def accuracy_report(X_train_scale, y_train,X_test_scale,y_test,criterion='gini',max_depth=4):
    tree = DecisionTreeClassifier(criterion = criterion, max_depth=max_depth)
    tree.fit(X_train_scale, y_train)
    pred = tree.predict(X_train_scale)
  
    print(criterion + " tree train accuracy: %f" 
        % (sklearn.metrics.accuracy_score(y_train, pred )))
    pred = tree.predict(X_test_scale)
    print(criterion + " tree test accuracy: %f" 
        % (sklearn.metrics.accuracy_score(y_test, pred )))
    

def main():
    X_train, X_test, y_train, y_test = load_train_test_data(test_ratio=.3,random_state=1)
    # print("a {0} \n b{1} \n c--- {2} \n d---{3}".format(len(X_train),len(X_test),len(y_train),len(y_test)))
    # print("a {0} \n b{1} \n c--- {2} \n d---{3}".format(X_train,X_test,y_train,y_test))
    X_train_scale, X_test_scale = scale_features(X_train, X_test)
    print("X_train_scale {0} \n X_test_scale{1}".format(sorted(X_train_scale[:,0]), X_train_scale))
    print('----------------')
    # gini tree
    accuracy_report(X_train_scale, y_train,X_test_scale,y_test,criterion='gini',max_depth=4)
    # entropy tree
    accuracy_report(X_train_scale, y_train,X_test_scale,y_test,criterion='entropy',max_depth=4) 
    

if __name__ == "__main__":
    main()
    

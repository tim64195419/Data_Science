from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics
import numpy as np
import math



def load_train_test_data(test_ratio=.3, random_state = 1):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    sc = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_ratio, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std , X_test_std

def test(y):
    num_samples_per_class = [np.sum(y == i) for i in range(3)]
    
    predicted_class = np.argmax(num_samples_per_class)
    print( 'test~~~',num_samples_per_class,predicted_class)

def _entropy(sample_y,n_classes):
        total_num_samples = sample_y.size
        num_samples_per_class = [np.sum(sample_y == i) for i in range(n_classes)]
        print('num_samples_per_class',num_samples_per_class)

        entropy = 0
        for i in num_samples_per_class:
            if i ==0:
                continue
            else:
                entropy = entropy + (i/total_num_samples)*np.log2(total_num_samples/i)

        print('entropy',entropy)
        return entropy

# Calculate the Gini index for a split dataset


def _gini(sample_y,n_classes):
    total_num_samples = sample_y.size
    num_samples_per_class = [np.sum(sample_y == i) for i in range(n_classes)]
    print('num_samples_per_class',num_samples_per_class)
    p = 0
    for i in num_samples_per_class:
        p = p+ (i/total_num_samples)**2
    gini = 1 - p 

    return gini



def main():
    X_train, X_test, y_train, y_test = load_train_test_data(test_ratio=.3,random_state=1)
    X_train_scale, X_test_scale = scale_features(X_train, X_test)
    # print("a {0} \n b{1} \n c--- {2} \n d---{3}".format(len(X_train),len(X_test),len(y_train),len(y_test)))
    print("a {0} \n b{1} \n c--- {2} \n d---{3}".format(X_train,X_test,y_train,y_test))
    print("a~~~ {0} \n b~~~{1} \n ".format(len(X_train_scale),len(X_test_scale)))
    print(np.random.seed(123))
    print('----------------')
    # print(_gini(y_train,3))
    # print(type(y_train),y_train.dtype)
    # print(_gini(np.array([ 0,1]),2))
    # print(_entropy(y_train,3))
    _entropy(np.array([ 1,1]),2)
    
    # print(gini_index())

    

if __name__ == "__main__":
    main()
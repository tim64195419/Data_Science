from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics
import numpy as np
import math
import pandas as pd


def load_train_test_data(test_ratio=.3, random_state = 1):
    balance_scale = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data",
            names=['Class Name', 'Left-Weigh', 'Left-Distance', 'Right-Weigh','Right-Distance'],header=None)
    
    class_le = LabelEncoder()
    balance_scale['Class Name'] = class_le.fit_transform(balance_scale['Class Name'].values)
    X = balance_scale.iloc[:,1:].values
    y = balance_scale['Class Name'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, random_state=random_state, stratify=y)
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
    elements,counts = np.unique(sample_y,return_counts = True)
    print('entropy elements,counts ',elements,counts )
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

# Calculate the Gini index for a split dataset



def _gini(sample_y,n_classes):
    total_num_sample = sample_y.size
    elements,counts = np.unique(sample_y,return_counts = True)
    print('gini elements,counts ',elements,counts)
    gini = 0
    for i in counts:
        gini = gini + (i/total_num_sample)*(1-i/total_num_sample)
    return gini

def sort(X_train_scale,y_train):
    information_gain = []
    for j in range(X_train_scale.shape[1]):
        
        tem_train_scale = X_train_scale
        index = np.argsort(tem_train_scale[:,j])
        sort_x = tem_train_scale[:,j][index]
        sort_y = y_train[index]
        find_midpoint = []
        print('~~~~~~~~~a',X_train_scale.shape[1])
        print(index)
        print('index~~~~~~~')
        print(sort_x)
        print('y_train')
        print(sort_y)
        print('~~~~~~~~~')
        # print(X_train_scale_sort,len(X_train_scale_sort))
        for i in range(len(index)-1):
            if(sort_x[i]<sort_x[i+1]):
                midpoint = (sort_x[i]+sort_x[i+1])/2
                print(sort_x[i],sort_x[i+1],midpoint)
                print(i,i+1)
                
                find_midpoint.append([midpoint,i])
        
        for i in range(len(find_midpoint)):
            left = sort_y[:find_midpoint[i][1]]
            right = sort_y[find_midpoint[i][1]:]
            information_gain.append([_entropy(left,3),_entropy(right,3),find_midpoint[i][1],j])
            
            
            
            # print(len(left),len(right))
    print('information_gain')
    print(information_gain)
    
    
    
        
        
        
        
    

def main():
    X_train, X_test, y_train, y_test = load_train_test_data(test_ratio=.3,random_state=1)
    X_train_scale, X_test_scale = scale_features(X_train, X_test)
    # print("a {0} \n b{1} \n c--- {2} \n d---{3}".format(len(X_train),len(X_test),len(y_train),len(y_test)))
    # print("a {0} \n b{1} \n c--- {2} \n d---{3}".format(X_train,X_test,y_train,y_test))
    # print("a~~~ {0} \n b~~~{1} \n ".format(len(X_train_scale),len(X_test_scale)))
    # print("a~~~ {0} \n b~~~{1} \n ".format(X_train_scale,X_test_scale))
    # print(np.random.seed(123))
    print('----------------')
    sort(X_train_scale,y_train)
    # print(_gini(y_train,3))
    # print(_entropy(y_train,3))
    # print(type(y_train),y_train.dtype)
    # print(_gini(np.array([ 0,1]),2))
    # print(_gini(np.array([ 0,0,0,0,0,0,0,0,0,2,2,2,2,2]),2))
    # print(_entropy(np.array([ 0,0,0,0,0,0,2,2,2,2]),3))


    
    # print(gini_index())

    

if __name__ == "__main__":
    main()
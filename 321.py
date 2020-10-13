from sklearn.preprocessing import StandardScaler
import numpy as np

def gini_index(groups, classes):
    n_instances=float(sum([len(group) for group in groups]))
    print('n_instances',n_instances,'\n')
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            
            p = [row[-1] for row in group].count(class_val) / size
            print('p',p)
            score += p * p
        
        gini += (1.0 - score) * (size / n_instances)


    
    return gini


if __name__ == '__main__':
    print(gini_index([[[1, 1,1], [1, -1,-1]], [[1, 1], [1, -1]]], [1, -1]))
    print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()



def iris_classifier(x_index, y_index):
    # this formatter will label the colorbar with the correct target names
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
    
    plt.figure()
    plt.title('x_index = %1.0f , y_index=%1.0f' % (x_index,y_index))
    plt.scatter(iris.data[:, x_index], iris.data[:, y_index], 
                c=iris.target, cmap=plt.cm.get_cmap('viridis', 3))

    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.clim(-0.5, 2.5)
    plt.xlabel(iris.feature_names[x_index])
    plt.ylabel(iris.feature_names[y_index])
    

index = [[0,1],[0,2],[1,2]]

for arg in index:
    iris_classifier(arg[0],arg[1])
    


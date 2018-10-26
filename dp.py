import sys
import pandas as pd
from sklearn import preprocessing,model_selection,neighbors
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

def generate_graph(graph,X_test,i,r):
    for k in range(len(X_test)):
        dis = np.linalg.norm(X_test[k]-X_test[i])
        # print(y_train[k])
        if(dis<=2*r):
            graph[k][i]=1
            graph[i][k]=1
#             count[y_train[k][0]]+=1
    return 1

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.2)

graph = []
[graph.append([0]*(len(X_test))) for i in range(len(X_test))]

for i in range(len(X_test)):
	generate_graph(graph,X_test,i,0.5)

for i in range(len(graph)):
	for j in range(len(graph[i])):
		print(graph[i][j],end = " ")
	print()
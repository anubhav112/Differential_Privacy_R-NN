import sys
import pandas as pd
from sklearn import preprocessing,model_selection,neighbors
import numpy as np
# import matplotlib.pyplot as plt
import subprocess

df = pd.read_csv('Titanic_train.csv')
df.drop(['PassengerId','Name','Ticket','Cabin'],1,inplace=True)
df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Embarked'].fillna('S',inplace=True)
df['Sex']=df['Sex'].replace({"male":100.0,
	"female":-100.0})
df['Embarked']=df['Embarked'].replace({"S":100.0,
	"C":200.0,
	"Q":300.0})

X_train=np.array(df.drop(['Survived'],1))
y_train=np.array(df[['Survived']])

X_train = (X_train - X_train.mean(axis = 0))/X_train.std(axis = 0)
X_train_set,X_test_set,y_train_set,y_test_set=model_selection.train_test_split(X_train,y_train,test_size=0.2)

def generate_graph(graph,X_test,i,r):
    for k in range(len(X_test)):
        dis = np.linalg.norm(X_test[k]-X_test[i])
        # print(y_train[k])
        if(dis<=2*r):
            graph[k][i]=1
            graph[i][k]=1
#             count[y_train[k][0]]+=1
    return 1

graph = []
vis = []
cur_comp = []
clique = ([0]*len(X_test_set))
[graph.append([0]*(len(X_test_set))) for i in range(len(X_test_set))]

for i in range(len(X_test_set)):
    generate_graph(graph,X_test_set,i,0.25)

for i in range(len(graph)):
	graph[i][i]=0

def dfs(cur) :
	vis.append(cur)
	cur_comp.append(cur)
	for i in range(len(graph[cur])):
		if graph[cur][i] == 1 and i not in vis :
			dfs(i)

cnt = 0

for i in range(len(X_test_set)) :
	if i not in vis:
		dfs(i)
		graph1 = []
		[graph1.append([0]*(len(cur_comp))) for i in range(len(cur_comp))]
		for j in range(len(cur_comp)):
			for k in range(len(cur_comp)):
				graph1[j][k] = graph[cur_comp[j]][cur_comp[k]]
		oldstdout = sys.stdout
		sys.stdout = open("input.txt", "w")
		print(len(graph1))
		for j in range(len(graph1)):
			for k in range(len(graph1[j])):
				print(graph1[j][k],end=" ")
		print()
		sys.stdout = oldstdout
		val = subprocess.call(["./maximum_clique", "input.txt"])
		for j in range(len(cur_comp)):
			clique[cur_comp[j]] = val
		cur_comp = []
		cnt += 1

def RNN_classifier(X_train,y_train,tst,r, epsilon):
	unique = np.unique(y_train)
	count = [0]*len(unique)
	for k in range(len(X_train)):
		dis = np.linalg.norm(X_train[k]-tst)
		if(dis<=r):
			count[y_train[k][0]]+=1
	for i in range(len(count)) :
		count[i] = count[i] + np.random.laplace(scale = 1.0/epsilon)
	return np.argmax(count)

number_of_tests = int(input("Enter number of Epsilons : "))
array_result = []
while number_of_tests > 0:
	epsilon = float(input("Enter Epsilon : "))
	cnt = 0
	for i in range(len(X_test_set)) :
		output = RNN_classifier(X_train_set, y_train_set, X_test_set[i], 0.25, epsilon/clique[i])
		if output == y_test_set[i] :
			cnt += 1
	array_result.append("\t\t\t" + str(epsilon) + "\t\t" + str(cnt/len(X_test_set)))
	# array_result.append(str(epsilon) + "\t" + str(cnt/len(X_test_set)))
	number_of_tests -= 1
# tmpout = sys.stdout
# sys.stdout = open("plot.txt", "w")
print("\n\t\t\tEpsilon\t\tAccuracy")
for i in array_result:
	print(i)
print()
epsilon = float(input("Enter Epsilon1 for analysis : "))
print(X_train_set[100]),print(y_train_set[100])
for i in range(10):
	output = RNN_classifier(X_train_set, y_train_set, X_train_set[100], 0, epsilon)
	print(output, end=" ")
print()
epsilon = float(input("Enter Epsilon2 for analysis : "))
print(X_train_set[100]),print(y_train_set[100])
for i in range(10):
	output = RNN_classifier(X_train_set, y_train_set, X_train_set[100], 0, epsilon)
	print(output, end=" ")
print()

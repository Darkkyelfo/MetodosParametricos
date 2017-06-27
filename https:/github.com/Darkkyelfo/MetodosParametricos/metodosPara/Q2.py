'''
Created on Jun 25, 2017

@author: raul
'''
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

#gerando a base

tamBase = 120

mCov = [[1.2,0.4],
        [0.4,1.8]]

m1 = [0.1,0.1]
m2 = [2.1,1.9]
m3 = [-1.5,2]

base = []
classes = []

x0,y0 = np.random.multivariate_normal(m1,mCov,int(tamBase/3)).T
x1,y1 = np.random.multivariate_normal(m2,mCov,int(tamBase/3)).T
x2,y2 = np.random.multivariate_normal(m3,mCov,int(tamBase/3)).T

for i in range(tamBase):
    if(i<tamBase/3):
        base.append([x0[i],y0[i]])
        classes.append("c1")
    elif(i<(tamBase*2)/3):
        base.append([x1[i-40],y1[i-40]])
        classes.append("c2")
    elif(i<(tamBase*3)/3):
        base.append([x2[i-80],y2[i-80]])
        classes.append("c3")

print(base)
kf = KFold(n_splits=10, random_state=None, shuffle=True)
knn = KNeighborsClassifier(n_neighbors=1)
gnd = GaussianNB()
erroKnn = 0
erroGnd = 0
base = np.array(base)

classes = np.array(classes)
for train_index, test_index in kf.split(base):
    X_train, X_test = base[train_index], base[test_index]
    y_train, y_test = classes[train_index], classes[test_index]
    #knn
    knn.fit(X_train , y_train)
    knnPredict = knn.predict(X_test )
    print("fold knn:", 1-accuracy_score(y_test,knnPredict))
    erroKnn = (1-accuracy_score(y_test,knnPredict)) + erroKnn
    #bayes
    gnd.fit(X_train , y_train)
    gndPredict = gnd.predict(X_test )
    print("fold gnd:", 1-accuracy_score(y_test,gndPredict))
    erroGnd = (1-accuracy_score(y_test,gndPredict))  + erroGnd
    print("\n")
    
print("Taxa de erro knn:",erroKnn/10)
print("Taxa de erro gnd:",erroGnd/10)
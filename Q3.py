'''
Created on 26 de jun de 2017

@author: raul
'''
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import math

def px(m,v,x):#m-> media, v-> variancia e x -> vetor
    return (1/(math.sqrt(2*v*math.pi)))*math.exp(-1*(1/(2*v))*np.dot((x-m).T,(x-m)))

def classificar(m1,m2,v1,v2,x_teste):
    classes = []
    for x in x_teste:
        if(px(m1,v1,x)>px(m2,v2,x)):
            classes.append(0)
        else:
            classes.append(1)
    return classes

m1 = [1,1]
m2 = [1.5,1.5]
m3 = [3,3]

v = 0.2

mCov = [[0.2,0],
        [0,0.2]]

x0,y0 = np.random.multivariate_normal(m1,mCov,100).T
x1,y1 = np.random.multivariate_normal(m2,mCov,100).T
x2,y2 = np.random.multivariate_normal(m3,mCov,100).T

base = []
classes = []

for i in range(len(x0)):
    base.append([x0[i],y0[i]])
    classes.append(0)
    
for i in range(len(x1)):
    base.append([x1[i],y1[i]])
    classes.append(1)


kf = KFold(n_splits=10, shuffle=True)
base = np.array(base)
classes = np.array(classes)
erro = 0

claPredita = []
for train_index, test_index in kf.split(base):
    X_train, X_test = base[train_index], base[test_index]
    y_train, y_test = classes[train_index], classes[test_index]
    claPredita = classificar(m1, m2, v, v, X_test)
    erro = (1-accuracy_score(y_test,claPredita)) + erro

print("Taxa de erro %s:%s"%([1.5,1.5],erro/10))


base = []
classes = []

for i in range(len(x0)):
    base.append([x0[i],y0[i]])
    classes.append(0)
    
for i in range(len(x2)):
    base.append([x2[i],y2[i]])
    classes.append(1)


kf = KFold(n_splits=10, shuffle=True)
base = np.array(base)
classes = np.array(classes)
erro = 0

claPredita = []
for train_index, test_index in kf.split(base):
    X_train, X_test = base[train_index], base[test_index]
    y_train, y_test = classes[train_index], classes[test_index]
    claPredita = classificar(m1, m3, v, v, X_test)
    erro = (1-accuracy_score(y_test,claPredita)) + erro

print("Taxa de erro %s:%s"%(m3,erro/10))





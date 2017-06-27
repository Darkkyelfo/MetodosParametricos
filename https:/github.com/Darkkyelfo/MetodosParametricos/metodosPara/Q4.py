'''
Created on 27 de jun de 2017

@author: raul
'''
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import math as m

def px(me,mv,x):#me-> media, mv-> matrix de covariancia e x -> vetor
    return (1/(2*m.pi*m.pow(np.linalg.det(mv), 1/2)))*m.exp(-1*1/2*(x-me).T.dot(np.linalg.inv(mv)).dot(x-me))

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

mCov = [[1.01,0.2],
      [0.2,1.01]]

base =[]
classes = []

x0,y0 = np.random.multivariate_normal(m1,mCov,100).T
x1,y1 = np.random.multivariate_normal(m2,mCov,100).T

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
    claPredita = classificar(m1, m2, mCov , mCov , X_test)
    erro = (1-accuracy_score(y_test,claPredita)) + erro
    
print(erro/10)


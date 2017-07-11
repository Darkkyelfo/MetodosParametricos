'''
Created on Jun 6, 2017

@author: raul
'''
import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


def pCondicional(pXcy,pClasse,pX):
    return float(pXcy * pClasse)/pX

def separarRegiao(pC1xy,pC2xy):
    r1 = []
    r2 = []
    for i,j in enumerate(pC1xy):
        if(pC1xy[i]>pC2xy[i]):
            r1.append(i)
        else:
            r2.append(i)
    return r1,r2

def calcularErro(r1,r2,pxC1,pxC2,pc1,pc2):
    erro1 = 0
    erro2 = 0
    for i in r2:
        erro1 += pxC1[i]
    for i in r1:
        erro2 +=pxC2[i] 
    erro = pc1*erro1 + pc2*erro2
    return erro

base = []
classes = []

pc1 = 0.4
pc2 = 0.6

p0c1 = 0.3#12
p1c1 = 0.4#16
p2c1 = 0.3#12
#24+16 = 40
p1c2 = 0.2 #12
p2c2 = 0.6#36
p3c2 = 0.2#12
#24+36 = 60
tamanho = 100

for i in range(tamanho):
    if i<tamanho*pc1:
        classes.append("c1")
    else:
        classes.append("c2")
for j in range(int(tamanho*pc1)):
    if(base.count(0)<classes.count("c1")*p0c1):
        base.append(0)
    elif(base.count(1)<classes.count("c1")*p1c1):
        base.append(1)
    else:
        base.append(2)
cont1 = base.count(1)
cont2 = base.count(2)
for k in range(int(tamanho*pc2)):
    if(base.count(1)<(classes.count("c2")*p1c2)+cont1):
        base.append(1)
    elif(base.count(2)<(classes.count("c2")*p2c2)+cont2):
        base.append(2)
    else:
        base.append(3)

print(classes.count("c1"),classes.count("c2"))
print(len(base),base.count(0),base.count(1),
      base.count(2),base.count(3))

#embaralhar mantendo a relacao entre as listas
#Questao 2

base = np.array(base)
classes = np.array(classes)

kf = KFold(n_splits=10, shuffle=True)
knn = KNeighborsClassifier(n_neighbors=1)
gnd = GaussianNB()
erroKnn = 0
erroGnd = 0
for train_index, test_index in kf.split(base):
    X_train, X_test = base[train_index], base[test_index]
    y_train, y_test = classes[train_index], classes[test_index]
    #knn
    knn.fit(X_train.reshape(-1, 1) , y_train.reshape(-1, 1) )
    knnPredict = knn.predict(X_test.reshape(-1, 1) )
    print("fold knn:", 1-accuracy_score(y_test,knnPredict))
    erroKnn = (1-accuracy_score(y_test,knnPredict)) + erroKnn
    #bayes
    gnd.fit(X_train.reshape(-1, 1) , y_train.reshape(-1, 1))
    gndPredict = gnd.predict(X_test.reshape(-1, 1) )
    print("fold gnd:", 1-accuracy_score(y_test,gndPredict))
    erroGnd = (1-accuracy_score(y_test,gndPredict))  + erroGnd
    print("\n")
pXc1 = [p0c1,p1c1,p2c1,0]
pXc2 = [0,p1c2,p2c2,p3c2]

pCond1 = {0:0,1:0,2:0,3:0}
pCond2 = {0:0,1:0,2:0,3:0}

for i,j in enumerate(pXc1):
    pCond1[i] = pCondicional(j, pc1,(pc1*pXc1[i]+pc2*pXc2[i]))

for i,j in enumerate(pXc2):
    pCond2[i] = pCondicional(j, pc2,(pc1*pXc1[i]+pc2*pXc2[i]))

print(pCond1)
print(pCond2)

r1,r2  = separarRegiao(pCond1, pCond2)

print("regiões r1:%s, r2:%s "%(r1,r2))

erro = calcularErro(r1, r2, pXc1, pXc2, pc1, pc2)

print("Erro da base:%s"%erro)
print("Taxa de erro knn:",erroKnn/10)
print("Taxa de erro gnd:",erroGnd/10)





    
    
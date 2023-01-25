# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:02:52 2018

@author: smrzh
"""


import numpy as np
from sklearn.ensemble import RandomForestClassifier

def readFromFile(fileName):
    trainData = []
    with open(fileName, "r") as filestream:
        for line in filestream:
            currentline = line.split(",")
            featureV = list(map(float, currentline))
            trainData.append(featureV)
    return trainData
            
            
        
X = readFromFile("G:\\Projects\\CoinPy\\features1.txt")        
#print(X)
y = [1,1,1,2,1,1,0,2,1,0,2,1,2,0,0,2]
#0 is quarter, 1 is loonie and 2 is toonie


#one out accuracy
numRight = 0
for i in range (0,len(y)):
    X_1 = list(X)
    y_1 = list(y)
    XI = X[i]
    yI = y[i]
    del X_1[i]
    del y_1[i]
    clf = RandomForestClassifier(max_depth=11)
    clf.fit(X_1, y_1)
    if(clf.predict([XI])[0] == yI):
        numRight = numRight + 1
    print(str(yI) + "->" + str(clf.predict([XI])[0]))
    
print (numRight)

import datetime
import numpy as np
import pandas as pd
from sklearn import metrics, neighbors
import time
print ("Loading training data...")

clicks = np.loadtxt("/Users/adminadmin/Documents/mywork/e-commerce/yoochoose-data/yoochoose-clickssave1.txt", dtype=np.float)

X = clicks[:1000000, 1:-1]
y = clicks[:1000000, -1]

X1 = clicks[9229729:9239729, 1:-1]
y1 = clicks[9229729:9239729, -1]
'''
print ("Fitting data...")
max_recall=0
optimize_k=0
for i in range(1,22,2):
    alg = neighbors.KNeighborsClassifier(n_neighbors=i)
    alg.fit(X, y)

    print("Calculating training results...")
    acc1=metrics.accuracy_score(y, alg.predict(X))
    recall1=metrics.confusion_matrix(y, alg.predict(X))
    recall10 = recall1[1][0]
    recall11 = recall1[1][1]
    recall_rate1 = recall11 / (recall10 + recall11)

    print("Calculating val results...")
    acc2=metrics.accuracy_score(y1, alg.predict(X1))
    recall2=metrics.confusion_matrix(y1, alg.predict(X1))
    recall20 = recall2[1][0]
    recall21 = recall2[1][1]
    recall_rate2 = recall21 / (recall20 + recall21)
    print (str(i)+" train "+str(acc1)+" "+str(recall_rate1))
    print(str(i) + " val " + str(acc2) + " " + str(recall_rate2))

    if (recall_rate2>max_recall):
        max_recall=recall_rate2
        optimize_k=i
print ("best k: ",optimize_k)
'''

alg = neighbors.KNeighborsClassifier(n_neighbors=1)
alg.fit(X, y)
print ("Calculating test results...")
now=time.time()
X2 = clicks[9239729:, 1:-1]
y2 = clicks[9239729:, -1]
print (metrics.accuracy_score(y2, alg.predict(X2)))
recall3=metrics.confusion_matrix(y2, alg.predict(X2))
recall30 = recall3[1][0]
recall31 = recall3[1][1]
recall_rate3 = recall31 / (recall30 + recall31)
print (recall_rate3)
print (recall3)
now1=time.time()
print (now1-now)

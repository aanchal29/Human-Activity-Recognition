# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 00:22:56 2017

@author: DELL LAPTOP
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pylab as plt

###############################################################################

def loadData(filename):
    X = np.loadtxt(filename)
    return X
    
X_train = loadData('C:/Users/DELL LAPTOP/Documents/MATLAB/HAR/UCI HAR Dataset/train/X_train.txt')
y_train = loadData('C:/Users/DELL LAPTOP/Documents/MATLAB/HAR/UCI HAR Dataset/train/y_train.txt')
X_test = loadData('C:/Users/DELL LAPTOP/Documents/MATLAB/HAR/UCI HAR Dataset/test/X_test.txt')
y_test = loadData('C:/Users/DELL LAPTOP/Documents/MATLAB/HAR/UCI HAR Dataset/test/y_test.txt')

###############################################################################

def total_count(y):
    count = np.zeros(6)
    for i in range(y.size):
        if (y[i] == 1):
            count[0] += 1
        if (y[i] == 2):
            count[1] += 1
        if (y[i] == 3):
            count[2] += 1
        if (y[i] == 4):
            count[3] += 1
        if (y[i] == 5):
            count[4] += 1
        if (y[i] == 6):
            count[5] += 1
    return count

y_test_count = total_count(y_test)

def confusion_matrix(y_test, y_pred):
    cm = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            for k in range(y_pred.size):
                if (y_pred[k] == j + 1 and y_test[k] == i + 1):
                    cm[i][j] += 1
    return cm


def pre_rec(cm, count):
    perc = []
    for i in range(6):
        perc.append(float("{0:.2f}".format(cm[i][i] / count[i] * 100)))
    return perc
        
def overall_accuracy(cm, y_test):
    sum = 0
    for i in range(6):
        sum += cm[i][i]
    return float("{0:.2f}".format(sum * 100.0 / y_test.size))    
        
###############################################################################
        
pca2 = PCA(n_components = 2)
pca2.fit(X_train)
X_train_2 = pca2.transform(X_train)
X_test_2 = pca2.transform(X_test)

x11 = []
x12 = []
x21 = []
x22 = []
x31 = []
x32 = []
x41 = []
x42 = []
x51 = []
x52 = []
x61 = []
x62 = []

for i in range(len(y_test)):
    if (y_test[i] == 1):
        x11.append(X_test_2[i][0])
        x12.append(X_test_2[i][1])
    elif (y_test[i] == 2):
        x21.append(X_test_2[i][0])
        x22.append(X_test_2[i][1])
    elif (y_test[i] == 3):
        x31.append(X_test_2[i][0])
        x32.append(X_test_2[i][1])
    elif (y_test[i] == 4):
        x41.append(X_test_2[i][0])
        x42.append(X_test_2[i][1])
    elif (y_test[i] == 5):
        x51.append(X_test_2[i][0])
        x52.append(X_test_2[i][1])
    else:
        x61.append(X_test_2[i][0])
        x62.append(X_test_2[i][1])

plt.figure() 
plt.plot(x41, x42, 'xr', label = 'Sitting')
plt.plot(x51, x52, 'xm', label = 'Standing')     
plt.plot(x11, x12, 'xc', label = 'Walking')
plt.plot(x21, x22, 'xb', label = 'Upstairs')
plt.plot(x31, x32, 'xy', label = 'Downstairs')
plt.plot(x61, x62, 'xg', label = 'Laying')
plt.legend(loc='lower left')
plt.show()

###############################################################################
        
SVMK = SVC(probability = True)
SVMK.fit(X_train, y_train)
y_predict_svmk = SVMK.predict(X_test)

y_pred_count_svmk = total_count(y_predict_svmk)
cmatrix_svmk = confusion_matrix(y_test, y_predict_svmk)

print "\nSVM with kernel = 'rbf':"
print cmatrix_svmk
print ""

recall_svmk = pre_rec(cmatrix_svmk, y_test_count)
precision_svmk = pre_rec(cmatrix_svmk, y_pred_count_svmk)
accuracy_svmk = overall_accuracy(cmatrix_svmk, y_test)
print "Precision for SVM: "
print precision_svmk
print "Recall for SVM: "
print recall_svmk
print "Accuracy for SVM: "
print accuracy_svmk

###############################################################################

SVMK2 = SVC(kernel='linear')
SVMK2.fit(X_train, y_train)
y_predict_svmk2 = SVMK2.predict(X_test)

y_pred_count_svmk2 = total_count(y_predict_svmk2)
cmatrix_svmk2 = confusion_matrix(y_test, y_predict_svmk2)

print "\nSVM with kernel = 'linear':"
print cmatrix_svmk2
print ""

recall_svmk2 = pre_rec(cmatrix_svmk2, y_test_count)
precision_svmk2 = pre_rec(cmatrix_svmk2, y_pred_count_svmk2)
accuracy_svmk2 = overall_accuracy(cmatrix_svmk2, y_test)

print "Precision for SVM: "
print precision_svmk2
print "Recall for SVM: "
print recall_svmk2
print "Accuracy for SVM: "
print accuracy_svmk2

###############################################################################

SVMK3 = SVC(kernel='poly')
SVMK3.fit(X_train, y_train)
y_predict_svmk3 = SVMK3.predict(X_test)

y_pred_count_svmk3 = total_count(y_predict_svmk3)
cmatrix_svmk3 = confusion_matrix(y_test, y_predict_svmk3)

print "\nSVM with kernel = 'poly'"
print cmatrix_svmk3
print ""

recall_svmk3 = pre_rec(cmatrix_svmk3, y_test_count)
precision_svmk3 = pre_rec(cmatrix_svmk3, y_pred_count_svmk2)
accuracy_svmk3 = overall_accuracy(cmatrix_svmk3, y_test)

print "Precision for SVM: "
print precision_svmk3
print "Recall for SVM: "
print recall_svmk3
print "Accuracy for SVM: "
print accuracy_svmk3
###############################################################################
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:12:40 2017

@author: DELL LAPTOP
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

###############################################################################
def loadData(filename):
    X = np.loadtxt(filename)
    return X
    
X_train = loadData('C:/Users/DELL LAPTOP/Documents/MATLAB/HAR/UCI HAR Dataset/train/X_train.txt')
y_train = loadData('C:/Users/DELL LAPTOP/Documents/MATLAB/HAR/UCI HAR Dataset/train/y_train.txt')
X_test = loadData('C:/Users/DELL LAPTOP/Documents/MATLAB/HAR/UCI HAR Dataset/test/X_test.txt')
y_test = loadData('C:/Users/DELL LAPTOP/Documents/MATLAB/HAR/UCI HAR Dataset/test/y_test.txt')

def error(y_test, y_predict, name):
    ercount = 0

    for p in range(len(y_test)):
        if (y_test[p] != y_predict[p]):
            ercount += 1
        
    test_rate = ercount * 1.0 / len(y_test)

    print "The test error rate of", name, "is", test_rate
    
def error6(y_test, y_predict, name):
    ercount = 0

    for p in range(len(y_test)):
        if (y_test[p] == 6 and y_predict[p] != 6):
            ercount += 1
        
    test_rate = ercount * 1.0 / len(y_test)

    print "The test error rate for 6 of", name, "is", test_rate
    print ""
    
###############################################################################
# 1 SVM with kernel = 'linear'    
SVMK = SVC(kernel='linear')
SVMK.fit(X_train, y_train)
y_predict_svmk = SVMK.predict(X_test)
error(y_test, y_predict_svmk, 'SVM with kernel = "linear"')
error6(y_test, y_predict_svmk, 'SVM with kernel = "linear"')

###############################################################################

# 3 LDA  
LDA = LDA()
LDA.fit(X_train, y_train)
y_predict_lda = LDA.predict(X_test)
error(y_test, y_predict_lda, 'LDA')
error6(y_test, y_predict_lda, 'LDA')
###############################################################################
# 5 Logistic Regression 
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_predict_lr = LR.predict(X_test)
error(y_test, y_predict_lr, 'Logistic Regression')
error6(y_test, y_predict_lr, 'Logistic Regression')

###############################################################################
# 6 K-Neighbors Classifier 
KNC = KNeighborsClassifier(8)
KNC.fit(X_train, y_train)
y_predict_knc = KNC.predict(X_test)
error(y_test, y_predict_knc, 'K-Neighbors Classifier')
error6(y_test, y_predict_knc, 'K-Neighbors Classifier')

###############################################################################

import numpy as np
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
        
#######LDA#####################################################################

lda = LDA()
lda.fit(X_train, y_train)
y_predict_lda = lda.predict(X_test)

y_pred_count_lda = total_count(y_predict_lda)
cmatrix_lda = confusion_matrix(y_test, y_predict_lda)

print "\nLDA:"
print cmatrix_lda
print ""

recall_lda = pre_rec(cmatrix_lda, y_test_count)
precision_lda = pre_rec(cmatrix_lda, y_pred_count_lda)
accuracy_lda = overall_accuracy(cmatrix_lda, y_test)

print "Precision for LDA: "
print precision_lda
print "Recall for LDA: "
print recall_lda
print "Accuracy for LDA: "
print accuracy_lda
                  
##############################Logistic Regression##############################
logr = LogisticRegression()
logr.fit(X_train, y_train)
y_predict_logr = logr.predict(X_test)

y_pred_count_logr = total_count(y_predict_logr)
cmatrix_logr = confusion_matrix(y_test, y_predict_logr)

print "\nLogistic Regression:"
print cmatrix_logr
print ""

recall_logr = pre_rec(cmatrix_logr, y_test_count)
precision_logr = pre_rec(cmatrix_logr, y_pred_count_logr)
accuracy_logr = overall_accuracy(cmatrix_logr, y_test)

print "Precision for LR: "
print precision_logr
print "Recall for LR: "
print recall_logr
print "Accuracy for LR: "
print accuracy_logr

####################K-Neighbors Classifier#####################################

knc = KNeighborsClassifier(8)
knc.fit(X_train, y_train)
y_predict_knc = knc.predict(X_test)

y_pred_count_knc = total_count(y_predict_knc)
cmatrix_knc = confusion_matrix(y_test, y_predict_knc)

print "\nK-Neighbors Classifier:"
print cmatrix_knc
print ""

recall_knc = pre_rec(cmatrix_knc, y_test_count)
precision_knc = pre_rec(cmatrix_knc, y_pred_count_knc)
accuracy_knc = overall_accuracy(cmatrix_knc, y_test)

print "Precision for KNN: "
print precision_knc
print "Recall for KNN: "
print recall_knc
print "Accuracy for KNN: "
print accuracy_knc

###############################################################################
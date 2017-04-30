#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

print 'features_train size: ' + str(len(features_train))
print 'features_test size: ' + str(len(features_test))
print 'labels_train size: ' + str(len(labels_train))
print 'labels_test size: ' + str(len(labels_test))

t0 = time()

from sklearn.svm import SVC
# c= 10 : accuracy : 0.616040955631
# c= 100 : accuracy : 0.616040955631
# c= 1000 : accuracy : 0.821387940842
# c= 10000 : accuracy : 0.892491467577
# c= 100000 : accuracy : 0.860068259386
# c= 1000000 : accuracy : 0.860068259386
svc = SVC(kernel = 'rbf', C = 10000)
# svc = SVC(kernel = 'linear')

print 'training...'
svc.fit(features_train, labels_train)
print 'finished training.'
print "training time:", round(time()-t0, 3), "s"

print 'predicting...'
t0 = time()
pred = svc.predict(features_test)
print 'finished predicting'
print "predicting time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)

print 'Accuracy = ' + str(accuracy)

print 'Prediction# for Chris: ' + str(sum(1 for p in pred if p == 1))

# print 'for 10: ' + str(features_test[10]) + ' => ' + str(pred[10])
# print 'for 26: ' + str(features_test[26]) + ' => ' + str(pred[26])
# print 'for 50: ' + str(features_test[50]) + ' => ' + str(pred[50])

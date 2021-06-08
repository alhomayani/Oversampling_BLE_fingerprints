import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.svm import SVC


os.chdir('/content/gdrive/My Drive/training_testing_data/')

test = pd.read_csv('test_data_rp_3.csv')
X_test = test.iloc[:, :-1]
X_test = X_test.values
Y_test = test.iloc[:, -1:]
Y_test = Y_test.values
Y_test = Y_test.reshape((Y_test.shape[0],))

clf = SVC()

train = pd.read_csv('train_data_rp_3_IMBALANCED.csv')
X_train = train.iloc[:, :-1]
X_train = X_train.values
Y_train = train.iloc[:, -1:]
Y_train = Y_train.values
Y_train = Y_train.reshape((Y_train.shape[0],))
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print("IMBALANCED:", metrics.classification_report(Y_test, Y_pred, digits=4))

train = pd.read_csv('train_data_rp_3_SMOTE.csv')
X_train = train.iloc[:, :-1]
X_train = X_train.values
Y_train = train.iloc[:, -1:]
Y_train = Y_train.values
Y_train = Y_train.reshape((Y_train.shape[0],))
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print("SMOTE:", metrics.classification_report(Y_test, Y_pred, digits=4))

train = pd.read_csv('train_data_rp_3_ADASYN.csv')
X_train = train.iloc[:, :-1]
X_train = X_train.values
Y_train = train.iloc[:, -1:]
Y_train = Y_train.values
Y_train = Y_train.reshape((Y_train.shape[0],))
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print("ADASYN:", metrics.classification_report(Y_test, Y_pred, digits=4))

train = pd.read_csv('train_data_rp_3_VAE.csv')
X_train = train.iloc[:, :-1]
X_train = X_train.values
Y_train = train.iloc[:, -1:]
Y_train = Y_train.values
Y_train = Y_train.reshape((Y_train.shape[0],))
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print("VAE:", metrics.classification_report(Y_test, Y_pred, digits=4))

train = pd.read_csv('train_data_rp_3_CVAE.csv')
X_train = train.iloc[:, :-1]
X_train = X_train.values
Y_train = train.iloc[:, -1:]
Y_train = Y_train.values
Y_train = Y_train.reshape((Y_train.shape[0],))
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print("CVAE:", metrics.classification_report(Y_test, Y_pred, digits=4))


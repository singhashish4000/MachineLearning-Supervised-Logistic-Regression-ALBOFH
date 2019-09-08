#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 12:48:17 2019

@author: ashish
"""



### Importing Libraries ###

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import random
import time

random.seed(100)

## Data Preprocessing 

dataset = pd.read_csv('financial_data.csv')

# Feature Engineering 

dataset = dataset.drop(columns = ['months_employed'])
dataset["personal_account_months"] = (dataset.personal_account_m + (dataset.personal_account_y * 12))
dataset = dataset.drop(columns = ["personal_account_m", "personal_account_y"])


# One Hot Encoding

dataset = pd.get_dummies(dataset)
dataset = dataset.drop(columns = ["pay_schedule_semi-monthly"])

# Removing Extra Columns

response = dataset["e_signed"]
users = dataset["entry_id"]
dataset = dataset.drop(columns = ["e_signed", "entry_id"])


# Splitting into Train and Test Set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, response, test_size = 0.2, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2


### Model Building ###

### Comparing Models ###

## Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, penalty = 'l1')
classifier.fit(X_train, y_train)

# PRedicting Test Set

y_pred = classifier.predict(X_test)

# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['Logistic Regression (Lasso)', acc, prec, rec, f1]], columns = ['Model', 'Accuracy', 'Precesion', 'Recall', 'F1 Score'])



## Support Vector Machine (Linear)

from sklearn.svm import SVC
classifier = SVC(random_state = 0, kernel='linear')
classifier.fit(X_train, y_train)

# PRedicting Test Set

y_pred = classifier.predict(X_test)

# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

svm_results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1]], columns = ['Model', 'Accuracy', 'Precesion', 'Recall', 'F1 Score'])

results = results.append(svm_results, ignore_index = True)


## Support Vector Machine (rbf)

from sklearn.svm import SVC
classifier = SVC(random_state = 0, kernel='rbf')
classifier.fit(X_train, y_train)

# PRedicting Test Set

y_pred = classifier.predict(X_test)

# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

svm_rbf_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]], columns = ['Model', 'Accuracy', 'Precesion', 'Recall', 'F1 Score'])

results = results.append(svm_rbf_results, ignore_index = True)



## Random Forest Model ()

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0, n_estimators=100, criterion='entropy')
classifier.fit(X_train, y_train)

# PRedicting Test Set

y_pred = classifier.predict(X_test)

# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

random_forest_results = pd.DataFrame([['RandomForest (n=100)', acc, prec, rec, f1]], columns = ['Model', 'Accuracy', 'Precesion', 'Recall', 'F1 Score'])

results = results.append(random_forest_results, ignore_index = True)


# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, 
                             X = X_train,
                             y = y_train, cv = 10)

accuracies.mean()
print("Random Forest Classifier Accuracy: %0.2f (+/- %0.2f)" % (accuracies.mean(), accuracies.std()*2))

### Parameter Tuning

# Applying Grid Search

# Round 1: Entropy
parameteres = {"max_depth": [3, None],
               "max_features": [1, 5, 10],
               "min_samples_split": [2, 5 ,10],
               "min_samples_leaf": [1, 5, 10],
               "bootstrap": [True, False],
               "criterion": ["entropy"]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameteres,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)

t0 = time.thread_time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.thread_time()
print("Took %0.2f seconds" % (t1-t0))


rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters


# Round 2: Entropy
parameteres = {"max_depth": [None],
               "max_features": [3, 5, 7],
               "min_samples_split": [8, 10, 12],
               "min_samples_leaf": [1, 2, 3],
               "bootstrap": [True],
               "criterion": ["entropy"]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameteres,
                           scoring = "accuracy",
                           cv = 5,
                           n_jobs = 1)

t0 = time.thread_time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.thread_time()
print("Took %0.2f seconds" % (t1-t0))


rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters



# PRedicting Test Set

y_pred = grid_search.predict(X_test)

# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

random_forest_results = pd.DataFrame([['RandomForest (n=100, GS*2 + Entropy)', acc, prec, rec, f1]], columns = ['Model', 'Accuracy', 'Precesion', 'Recall', 'F1 Score'])

results = results.append(random_forest_results, ignore_index = True)

results

### End of Results ###

# Formatting Final Results

final_results = pd.concat([y_test, users], axis=1).dropna()
final_results['predictions'] = y_pred
final_results


























































































































#!/usr/bin/python

'''
D Hack Analyticsvidya 11/20/2015
'''

from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
import math

# Read train data

train = pd.read_csv('train.csv')
#Save all variables into featureNames except Product_ID and Purchase
featureNames = train.columns[1:-1]

features_train = train .drop('User_ID',1)
features_train = features_train.drop('Purchase',1)
labels_train = train['Purchase'].values

#Read test data
test = pd.read_csv('test.csv')
ids = test['User_ID'].values
pid = test['Product_ID'].values
features_test = test.drop('User_ID',1)





le = LabelEncoder()
print("assuming text variables are categorical & replacing them with numeric ids\n")
for c in featureNames:
   
   if features_train[c].dtype.name == 'object':
      
      le.fit(np.append(features_train[c],features_test[c])) 
      features_train[c] = le.transform(features_train[c]).astype(int)
      features_test[c] = le.transform(features_test[c]).astype(int)

features_train = features_train.fillna(0)
features_test = features_test.fillna(0)
print features_train
print labels_train
dhackModel = RandomForestRegressor(min_samples_split = 20)
dhackModel.fit(features_train,labels_train)
pred = dhackModel.predict(features_test)
print features_test

submission = pd.DataFrame({"User_ID" : ids, "Product_ID" : pid, "Purchase" : pred})
#print out the value of alpha that minimizes the CV-error
#print("alpha Value that Minimizes CV Error ",dhackLassoModel.alpha_)
#print("Minimum MSE ", min(dhackLassoModel.mse_path_.mean(axis=-1)))

submission.to_csv("submission.csv", index=False)



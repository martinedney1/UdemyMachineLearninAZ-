# -*- coding: utf-8 -*-

# Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values      # Locate position 
Y = dataset.iloc[:,3].values        # Locate position

# Taking care of the missing data
from sklearn.preprocessing import Imputer # Data mining library which is opensource
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder() # create an object to class labelencoder
X[:,0] = labelencoder_X.fit_transform(X[:,0]) # Selection column 0 i.e first encode nito intefers
onehotencoder = OneHotEncoder(categorical_features = [0]) # Specify firt column to be treated as cateforial features
X = onehotencoder.fit_transform(X).toarray() # Transform integers to binary column for algorithm
labelencoder_Y = LabelEncoder() # create an object to class labelencoder
Y = labelencoder_X.fit_transform(Y) # Selection rows and encode

#Splitting the data into the training set and data set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0) # Create split test and train data, define % of test size 

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

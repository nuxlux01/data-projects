# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 15:12:22 2020

@author: Ulises
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#importing the model to use
from sklearn.naive_bayes import GaussianNB
import seaborn as sns


#import dataset
iris = sns.load_dataset('iris')
print(iris.head())
xIris = iris.iloc[:,:3]
yIris = iris.iloc[:,4]
print(xIris.head())
print(yIris.head())
#separate train set and evaluation set
xTrain, xTest,yTrain, yTest = train_test_split(xIris, yIris, random_state = 1)

#instantiate the model
model = GaussianNB()

#fitting or training the model
model.fit(xTrain,yTrain)

#predict values from the model
yModel = model.predict(xTest)

#evaluate the model
print(accuracy_score(yTest,yModel))
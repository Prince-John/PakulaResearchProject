# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:38:11 2020

@author: Prince John
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
#Importing Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor

file = open(r'C:\Users\Prince John\Documents\Summer 2020\Research Project\datasets\labels.txt',  encoding='utf-8')
complete_label_list = [line.strip() for line in file.readlines()]



hungarian_raw=pd.read_csv(r'C:\Users\Prince John\Documents\Summer 2020\Research Project\datasets\CleanedData\cleaned_hungarian.csv')



def label_and_clean(data, feature_Set):
    data = data.drop(axis=1, labels = "Unnamed: 0")
    data.columns = complete_label_list
    
    return data

hungarian_labeled = label_and_clean(hungarian_raw,complete_label_list)

Y = hungarian_labeled[['58 num: diagnosis of heart disease (angiographic disease status)']]
X = hungarian_labeled.drop(axis = 1, labels = "58 num: diagnosis of heart disease (angiographic disease status)")



#Testing out how accurate the highly corelated cardiac artearies are at prediction the class
X_arteries = X[complete_label_list[58:67]] 
X_arteries = X_arteries.replace(to_replace = -9, value = 0)




   
X_embedded = TSNE(n_components=3).fit_transform(X)


plt.scatter( X_embedded[:,0],X_embedded[:,1], X_embedded[:,2])
plt.show()       
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 08:47:52 2020
This is the correlation analysis of lmt artery and svm, ann and desision tree analysis
@author: Prince John
"""


#import pandas_bokeh
#from bokeh.io import output_notebook
#from bokeh.plotting import figure, show

import pandas as pd
#pd.options.plotting.backend = 'pandas_bokeh'
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
from impyute.imputation.cs import fast_knn
from sklearn.preprocessing import OneHotEncoder

file = open(r'C:\Users\Prince John\Documents\Summer 2020\Research Project\PakulaResearchProject\datasets\labels.txt',  encoding='utf-8')
complete_label_list = [line.strip() for line in file.readlines()]



hungarian_raw=pd.read_csv(r'C:\Users\Prince John\Documents\Summer 2020\Research Project\PakulaResearchProject\datasets\CleanedData\cleaned_hungarian.csv')

remove = complete_label_list[0:2]+complete_label_list[19:22]+complete_label_list[35:36]+complete_label_list[44:46]+complete_label_list[51:57]+complete_label_list[68:75]

def label_and_clean(data, feature_Set):
    data = data.drop(axis=1, labels = "Unnamed: 0")
    data.columns = complete_label_list
    data = data.drop(axis=1, labels = remove)
    data = data.replace(-9, np.NaN)
   
    
    return data

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements


hungarian_labeled = label_and_clean(hungarian_raw,complete_label_list)


artery_accuracy = pd.DataFrame(columns = ["ANN", "SVM", "Tree"])
artery_prediction = pd.DataFrame(columns = ["ANN", "SVM", "Tree"])
artery_classifier = pd.DataFrame(columns = ["ANN", "SVM", "Tree"])


#remove the arteries and 58
X = hungarian_labeled.drop(axis = 1, labels = "58 num: diagnosis of heart disease (angiographic disease status)")
X = X.drop(axis=1, labels= complete_label_list[58:68])
print(X.info())
X=X.dropna(axis =1, thresh = 50)

Y_arteries = hungarian_labeled[complete_label_list[58:68]] 
Y_arteries = Y_arteries.replace(to_replace = np.NAN, value = 0)

X = pd.concat([X, Y_arteries], axis = 1)

#impute the data 
X=fast_knn(X.values.astype(float), k=30)


    
#spliting the data into test and train sets
train, test, Y_train, Y_test = train_test_split(X,hungarian_labeled[complete_label_list[57]], test_size = 0.2, stratify = hungarian_labeled[complete_label_list[57]])
    
X_train = train.T[0:34].T
X_test = test.T[0:34].T
Y_artery_train = train.T[34:]
Y_artery_test =  test.T[34:].T

for artery in Y_artery_train:
    
    
    #scaling the data
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    
    #ann = 
    clf_ANN = MLPClassifier(solver='lbfgs', activation = 'logistic', hidden_layer_sizes = (500))
    clf_ANN.fit(X_train, artery)
    
    #tree
    clf_gini = tree.DecisionTreeClassifier(criterion='gini')
    clf_gini.fit(X_train, artery)
    
    
    #SVM
    clf_SVM= svm.SVC(kernel = 'linear')
    clf_SVM.fit(X_train,  artery)
    
    #storing the classifiers
    classifiers = [clf_ANN, clf_gini, clf_SVM]
    artery_classifier.loc[len(artery_prediction)]  = classifiers
    
    
    
    Y_pred_ANN = clf_ANN.predict(X_test)
    Y_pred_tree = clf_gini.predict(X_test)
    Y_pred_SVM = clf_SVM.predict(X_test)
    
    predictions = [Y_pred_ANN, Y_pred_tree, Y_pred_SVM]
    artery_prediction.loc[len(artery_prediction)]  = predictions
    
    
    
    """
    cm_ANN = confusion_matrix(Y_pred_ANN, Y_artery_test)
    cm_tree = confusion_matrix(Y_pred_tree, Y_artery_test)
    cm_SVM = confusion_matrix(Y_pred_SVM, Y_test)
   
    
    #store the efficiency in the dataframe
    accuracies=[accuracy(cm_ANN), accuracy(cm_SVM), accuracy(cm_tree)]
    artery_accuracy.loc[len(artery_accuracy)]  = accuracies
     """












Y = hungarian_labeled[['58 num: diagnosis of heart disease (angiographic disease status)']]
X = hungarian_labeled.drop(axis = 1, labels = "58 num: diagnosis of heart disease (angiographic disease status)")





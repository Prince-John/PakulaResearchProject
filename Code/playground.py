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

hungarian_labeled = label_and_clean(hungarian_raw,complete_label_list)

#hungarian_




Y = hungarian_labeled[['58 num: diagnosis of heart disease (angiographic disease status)']]
X = hungarian_labeled.drop(axis = 1, labels = "58 num: diagnosis of heart disease (angiographic disease status)")


corr = hungarian_labeled.corr()[Y]



"""
# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
                                    'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)

"""




















#Testing out how accurate the highly corelated cardiac artearies are at prediction the class
X_arteries = X[complete_label_list[58:67]] 
X_arteries = X_arteries.replace(to_replace = np.NAN, value = 0)




   
#X_embedded = TSNE(n_components=3).fit_transform(X)


#plt.scatter( X_embedded[:,0],X_embedded[:,1], X_embedded[:,2])
#plt.show()       
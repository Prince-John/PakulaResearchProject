#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from impyute.imputation.cs import fast_knn
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor,  RandomForestClassifier


# In[11]:


file = open(r'labels.txt',  encoding='utf-8')
complete_label_list = [line.strip() for line in file.readlines()]
remove = complete_label_list[0:2]+complete_label_list[19:22]+complete_label_list[35:36]+complete_label_list[44:46]+complete_label_list[51:57]+complete_label_list[68:75]
# switzerland_raw= pd.read_csv('switzerland.csv')


# In[12]:


print(complete_label_list)


# In[13]:


switzerland_raw=pd.read_csv(r'switzerland.csv')


# In[25]:





# In[26]:


def label_and_clean(data, feature_Set):
    data = data.drop(axis=1, labels = "Unnamed: 0")
    data.columns = complete_label_list
    data = data.drop(axis=1, labels = remove)
    data = data.replace(-9, np.NaN)
    return data
switzerland_labeled = label_and_clean(switzerland_raw,complete_label_list)
switzerland_labeled.head()


# In[80]:


X = switzerland_labeled.drop(axis = 1, labels = "58 num: diagnosis of heart disease (angiographic disease status)")
X = X.drop(axis=1, labels= complete_label_list[58:68])
X=X.dropna(axis =1, thresh = 50)


# In[81]:


Y_arteries = switzerland_labeled[complete_label_list[58:68]] 
Y_arteries = Y_arteries.replace(to_replace = np.NAN, value = 0)

def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

artery_accuracy = pd.DataFrame(columns = ["ANN", "SVM", "Tree"])

def likelihood(top, bottom):
    if top==0:
        top = 1
    return top/bottom

def recall(confusion_matrix):
    column_sum = np.sum(confusion_matrix, axis=0)
    num=0
    while num < len(confusion_matrix.T):
        conditional_probability=confusion_matrix[num][num]/confusion_matrix.sum()
        conditional_probability=conditional_probability*likelihood(confusion_matrix[num][num],column_sum[num])
        num=num+1
    return conditional_probability
artery_recall =pd.DataFrame(columns = ["ANN", "SVM", "Tree"])


# In[82]:


X=fast_knn(X.values.astype(float), k=30) 


# In[ ]:





# In[83]:


for artery in Y_arteries.columns:
    
    #spliting the data into test and train sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_arteries[artery], test_size = 0.2, stratify = Y_arteries[artery])
    
    
    #scaling the data
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
     #ann = 
    ANN_no_PCA = MLPClassifier(solver='lbfgs', activation = 'logistic', hidden_layer_sizes = (500))
    ANN_no_PCA.fit(X_train, Y_train.to_numpy().flatten())
    
    #tree
    clf_gini = tree.DecisionTreeClassifier(criterion='gini')
    clf_gini.fit(X_train, Y_train)
    
    
    #SVM
    clf_SVM= svm.SVC(kernel = 'linear')
    clf_SVM.fit(X_train, Y_train)
    
    
    
    Y_pred_ANN = ANN_no_PCA.predict(X_test)
    Y_pred_tree = clf_gini.predict(X_test)
    Y_pred_SVM = clf_SVM.predict(X_test)
    cm_ANN = confusion_matrix(Y_pred_ANN, Y_test)
    cm_tree = confusion_matrix(Y_pred_tree, Y_test)
    cm_SVM = confusion_matrix(Y_pred_SVM, Y_test)
    #store the efficiency in the dataframe
    accuracies=[accuracy(cm_ANN), accuracy(cm_SVM), accuracy(cm_tree)]
    recall_value = [recall(cm_ANN), recall(cm_tree), recall(cm_SVM)]
        
    artery_recall.loc[len(artery_recall)]  = recall_value
    artery_accuracy.loc[len(artery_accuracy)]  = accuracies


# In[84]:


arr= [np.array(artery_recall.idxmax(axis=1)),np.array(artery_accuracy.idxmax(axis=1))]
print(arr)


# In[73]:


clf_RF = RandomForestClassifier(max_depth=10000, random_state=5)
clf_RF.fit(Y_artery_train.T, Y_train)


# In[ ]:


clf_RF_ext = RandomForestClassifier(max_depth=10000, random_state=5)
clf_RF_ext.fit(train, Y_train)


# In[ ]:


clf_RF = RandomForestClassifier(max_depth=10000, random_state=5)
clf_RF.fit(Y_artery_train.T, Y_train)


# In[ ]:


print(artery_recall)


# In[ ]:





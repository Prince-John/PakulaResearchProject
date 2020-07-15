#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from impyute.imputation.cs import fast_knn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

#models
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor,  RandomForestClassifier

#Importing Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE


# In[2]:


file = open(r'labels.txt',  encoding='utf-8')
complete_label_list = [line.strip() for line in file.readlines()]


# In[3]:


Switzerland_labeled= pd.read_csv('cleanedSwitzerland.csv')


# In[4]:


X = Switzerland_labeled.drop(axis = 1, labels = "58 num: diagnosis of heart disease (angiographic disease status)")
X = X.drop(axis=1, labels= complete_label_list[58:68])
X=X.dropna(axis =1, thresh = 50)

Y_arteries = Switzerland_labeled[complete_label_list[58:68]] 
Y_arteries = Y_arteries.replace(to_replace = np.NAN, value = 0)

Y_target=Switzerland_labeled[complete_label_list[57]]


# In[5]:


#not in use
def likelihood(top, bottom):
    if bottom==0:
        return 1
    return (top+1)/bottom

def recall(confusion_matrix):
    column_sum = np.sum(confusion_matrix, axis=0)
    num=0
    conditional_probability=1
    while num < len(confusion_matrix.T):
        conditional_probability=conditional_probability*(confusion_matrix[num][num]+1)/confusion_matrix.sum()*likelihood(confusion_matrix[num][num],column_sum[num])
        num=num+1
    return conditional_probability
artery_recall =pd.DataFrame(columns = ["ANN", "SVM", "Tree", "NB"])


# In[6]:


X=fast_knn(X.values.astype(float), k=30) 


# In[7]:


pipeline_SVM= Pipeline([('scalar1',StandardScaler()), ('pca1', PCA(n_components=2)),('clf_SVM',SVC(kernel = 'linear'))])


# In[8]:


pipeline_ANN= Pipeline([('scalar2',StandardScaler()), ('pca2', PCA(n_components=2)),('ANN',MLPClassifier(solver='lbfgs', activation = 'logistic', hidden_layer_sizes = (500)))])


# In[9]:


pipeline_DT= Pipeline([('scalar3',StandardScaler()), ('pca3', PCA(n_components=2)),('clf_gini',DecisionTreeClassifier(criterion='gini'))])


# In[10]:


pipeline_NB= Pipeline([('scalar4',StandardScaler()), ('pca4', PCA(n_components=2)),('clf_NB',GaussianNB())])


# In[11]:


pipelines = [pipeline_SVM, pipeline_ANN, pipeline_DT, pipeline_NB]
pipeline_dict= {0:"SVM",1:"ANN",2:"DT",3:"NB"}
print(Y_arteries.columns)


# In[12]:


arr=[]


# In[13]:


for artery in Y_arteries.columns:
    #norm = 
    best_accuracy=0.0
    best_classifier=0
    best_pipeline=""
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_arteries[artery], test_size = 0.2, stratify = Y_arteries[artery])
    
    for pipe in pipelines:
        pipe.fit(X_train, Y_train)
        
    #checking accuracy
    #for i,model in enumerate(pipelines):
    #   print("{} Test Accuracy: {}".format(pipeline_dict[i], model.score(X_test,Y_test)))
    
    for i,model in enumerate(pipelines):
        if model.score(X_test,Y_test)> best_accuracy:
            best_accuracy=model.score(X_test,Y_test)
            best_pipeline=model
            best_classifier =i 
    arr.append(format(pipeline_dict[best_classifier]))
    
    predicted = pipelines[i].predict(X)
    df= pd.read_csv('recalling.csv')
    df[artery]= predicted 


# In[14]:


#training for second tier
pipeline_RF=Pipeline([('scalar5',StandardScaler()), ('pca5', PCA(n_components=2)),('clf_Rf',RandomForestClassifier(max_depth=10000, random_state=5))])


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(df, Y_target, test_size = 0.2, stratify = Y_target)


# In[20]:


pipeline_RF.fit(X_train, Y_train)


# In[21]:


print(model.score(X_test,Y_test))


# In[ ]:





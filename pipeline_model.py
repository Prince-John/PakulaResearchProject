#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from impyute.imputation.cs import fast_knn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

#models
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# In[2]:


file = open(r'labels.txt',  encoding='utf-8')
complete_label_list = [line.strip() for line in file.readlines()]


# In[3]:


Switzerland_labeled= pd.read_csv('cleanedHungarian.csv')
Switzerland_labeled.shape


# In[4]:


X = Switzerland_labeled.drop(axis = 1, labels = "58 num: diagnosis of heart disease (angiographic disease status)")
X = X.drop(axis=1, labels= complete_label_list[58:68])
X=X.dropna(axis =1, thresh = 50)

Y_arteries = Switzerland_labeled[complete_label_list[58:68]] 
Y_arteries = Y_arteries.replace(to_replace = np.NAN, value = 0)

Y_target=Switzerland_labeled[complete_label_list[57]]
# Y_target=Y_target.replace(2,1)
# Y_target=Y_target.replace(3,1)
# Y_target=Y_target.replace(4,1)


# In[5]:


#separating numerical and categorical
columns=[]
categorical_features=[]
numeric_features=[]
columns=X.columns
for ho in columns:
    if "1 =" in ho:
        categorical_features.append(ho)
    else:
        numeric_features.append(ho)


# In[6]:


# numeric_features=['3 age: age in years','10 trestbps: resting blood pressure (in mm Hg on admission to the hospital)','12 chol: serum cholestoral in mg/dl','19 restecg: resting electrocardiographic results','29 thaldur: duration of exercise test in minutes','30 thaltime: time when ST measure depression was noted','31 met: mets achieved','32 thalach: maximum heart rate achieved','33 thalrest: resting heart rate','34 tpeakbps: peak exercise blood pressure (first of 2 parts)','35 tpeakbpd: peak exercise blood pressure (second of 2 parts)','37 trestbpd: resting blood pressure']
# categorical_features=['4 sex: sex (1 = male; 0 = female)','11 htn','5 painloc: chest pain location (1 = substernal; 0 = otherwise)','6 painexer (1 = provoked by exertion; 0 = otherwise)','7 relrest (1 = relieved after rest; 0 = otherwise)','23 dig (digitalis used furing exercise ECG: 1 = yes; 0 = no)','24 prop (Beta blocker used during exercise ECG: 1 = yes; 0 = no)','25 nitr (nitrates used during exercise ECG: 1 = yes; 0 = no)','26 pro (calcium channel blocker used during exercise ECG: 1 = yes; 0 = no)','27 diuretic (diuretic used used during exercise ECG: 1 = yes; 0 = no)','9 cp: chest pain type','38 exang: exercise induced angina (1 = yes; 0 = no)','39 xhypo: (1 = yes; 0 = no)']


# In[7]:


numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),('scaler', StandardScaler())])
categorical_transformer= Pipeline(steps=[('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])


# In[8]:


preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),('cat', categorical_transformer, categorical_features)])


# In[9]:


# #not in use
# def likelihood(top, bottom):
#     if bottom==0:
#         return 1
#     return (top+1)/bottom

# def recall(confusion_matrix):
#     column_sum = np.sum(confusion_matrix, axis=0)
#     num=0
#     conditional_probability=1
#     while num < len(confusion_matrix.T):
#         conditional_probability=conditional_probability*(confusion_matrix[num][num]+1)/confusion_matrix.sum()*likelihood(confusion_matrix[num][num],column_sum[num])
#         num=num+1
#     return conditional_probability
# artery_recall =pd.DataFrame(columns = ["ANN", "SVM", "Tree", "NB"])


# In[10]:


pipeline_SVM= Pipeline(steps=[('preprocessor', preprocessor), ('pca1', PCA(n_components=2)),('clf_SVM',SVC(kernel = 'linear'))])


# In[11]:


pipeline_ANN= Pipeline(steps=[('preprocessor', preprocessor), ('pca2', PCA(n_components=2)),('ANN',MLPClassifier(solver='lbfgs', activation = 'logistic', hidden_layer_sizes = (500)))])


# In[12]:


pipeline_DT= Pipeline(steps=[('preprocessor', preprocessor), ('pca3', PCA(n_components=2)),('clf_gini',DecisionTreeClassifier(criterion='gini'))])


# In[13]:


pipeline_NB= Pipeline(steps=[('preprocessor', preprocessor), ('pca4', PCA(n_components=2)),('clf_NB',GaussianNB())])


# In[14]:


pipelines = [pipeline_SVM, pipeline_ANN, pipeline_DT, pipeline_NB]
pipeline_dict= {0:"SVM",1:"ANN",2:"DT",3:"NB"}
df=pd.DataFrame()


# In[16]:


arr=[]
for artery in Y_arteries.columns:
    #norm = 
    best_accuracy=0.0
    best_classifier=0
    best_pipeline=""
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_arteries[artery], test_size = 0.2, stratify = Y_arteries[artery])
    
    for pipe in pipelines:
        pipe.fit(X_train, Y_train)
        
    #checking accuracy
    for i,model in enumerate(pipelines):
       print("{} Test Accuracy: {}".format(pipeline_dict[i], model.score(X_test,Y_test)))
    #finiding the best
    for i,model in enumerate(pipelines):
        if model.score(X_test,Y_test)> best_accuracy:
            best_accuracy=model.score(X_test,Y_test)
            best_pipeline=model
            best_classifier =i
    arr.append(format(pipeline_dict[best_classifier]))
    
    predicted = pipelines[i].predict(X)
    print(len(predicted))
    df= pd.read_csv('recalling.csv')
    df[artery]= predicted
    print("\n")
df.columns


# In[ ]:


#Experiment
#df.drop(columns=['60 ladprox', '67 rcaprox'])


# In[ ]:


#training for second tier
X_train, X_test, Y_train, Y_test = train_test_split(df, Y_arteries, test_size = 0.3, stratify = Y_target)
pipeline_RF_PCA=Pipeline([('scalar5',StandardScaler()), ('pca5', PCA(n_components=4)),('clf_Rf',RandomForestClassifier(max_depth=10000, random_state=5))])


# In[ ]:


clf_RF_ext = RandomForestClassifier(max_depth=10000, random_state=5)
clf_RF_ext.fit(X_train, Y_train)


# In[ ]:


pipeline_RF_PCA.fit(X_train, Y_train)


# In[ ]:


print(pipeline_RF_PCA.score(X_test,Y_test))
print(clf_RF_ext.score(X_test,Y_test))


# In[ ]:


cross_val_score(pipeline_RF_PCA,df, Y_target, cv= 5, scoring ='accuracy').mean()


# In[ ]:


cross_val_score(clf_RF_ext,df, Y_target, cv= 5, scoring ='accuracy').mean()


# In[ ]:


Y_target.value_counts(normalize=True)


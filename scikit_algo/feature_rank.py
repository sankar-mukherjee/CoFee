# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:10:14 2015

@author: mukherjee
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.learning_curve import learning_curve

# read Form data
DATA_FORM_FILE = 'all-merged-cat.csv'
#rawdata = pd.read_csv(DATA_FORM_FILE, usecols=np.r_[3,5:12,13:28,81:87,108])
rawdata = pd.read_csv(DATA_FORM_FILE)

#select features
posfeat = pd.DataFrame.as_matrix(rawdata)[:,np.r_[3:12]].astype(float)
lextypefeat = pd.DataFrame.as_matrix(rawdata)[:,np.r_[12:14]]
lexfeat = pd.DataFrame.as_matrix(rawdata)[:,np.r_[14:29]].astype(float)
phonfeat = pd.DataFrame.as_matrix(rawdata)[:,np.r_[29:47]]
accoufeat = pd.DataFrame.as_matrix(rawdata)[:,np.r_[47:81]].astype(float)

phonfeat = pd.DataFrame.as_matrix(rawdata)[:,np.r_[29]].astype(float)
lextypefeat = pd.DataFrame.as_matrix(rawdata)[:,np.r_[13]]
# Transforming categorical feature
le = preprocessing.LabelBinarizer()
le.fit(lextypefeat)
list(le.classes_)
lextypefeat = le.transform(lextypefeat)
#----------------------------------------------------------------------------------------------------
# select feature combination
featN = np.column_stack((posfeat,accoufeat,phonfeat))

featB = np.column_stack((lexfeat,lextypefeat))
###------------------------------------------- PCA
#from sklearn.decomposition import PCA
#pca = PCA(n_components=4)
#####------------------------------------------- Randomized PCA
##from sklearn.decomposition import RandomizedPCA
##pca = RandomizedPCA(n_components=30, whiten=True)
###
#scale = pca.fit(feat1)
#feat1 = scale.fit_transform(feat1)

feat = np.column_stack((featN,featB))
feat[np.isnan(feat)] = 0
feat[np.isinf(feat)] = 0
# select test labels
#Ytest = pd.DataFrame.as_matrix(rawdata)[:,20:26].astype(float)
label = pd.DataFrame.as_matrix(rawdata)[:,108]

#remove bad features as there is no label
scale = np.where(label == 'None')
label = np.delete(label,scale)
feat = np.delete(feat,scale,0)
#----------------------------------------------------------------------------------------------------
# Transforming categorical feature
le = preprocessing.LabelEncoder()
le.fit(label)
list(le.classes_)
label = le.transform(label)

# create traning and test data by partioning
train_set_size = int(.7*len(feat))
test_set_size = int(.3*len(feat))

XtrainPos = feat[0:train_set_size,:]
YtrainPos = label[0:train_set_size]

XtestPos = feat[train_set_size:,:]
YtestPos = label[train_set_size:]    

XtrainAll = feat           
#----------------------------------------------------------------------------------------------------
#normalization of features
scale = preprocessing.StandardScaler().fit(XtrainPos)
XtrainPos = scale.transform(XtrainPos)
XtestPos = scale.transform(XtestPos)
# for whole data set
scaleAll = preprocessing.StandardScaler().fit(XtrainAll)
XtrainAll = scaleAll.transform(XtrainAll)

#scale = preprocessing.MinMaxScaler()
#XtrainPos = scale.fit_transform(XtrainPos)
#XtestPos = scale.transform(XtestPos)
#scaleAll = preprocessing.MinMaxScaler()
#XtrainAll = scaleAll.fit_transform(XtrainAll)

#scale = preprocessing.Normalizer().fit(XtrainPos)
#XtrainPos = scale.transform(XtrainPos)
#XtestPos = scale.transform(XtestPos)
#scaleAll = preprocessing.Normalizer().fit(XtrainAll)
#XtrainAll = scaleAll.transform(XtrainAll)

###------------------------------------------- RandomizedLogisticRegression
#from sklearn.linear_model import RandomizedLogisticRegression
#scale = RandomizedLogisticRegression()
#XtrainPos = scale.fit_transform(XtrainPos,YtrainPos)
#XtestPos = scale.transform(XtestPos)
#XtrainAll = scale.fit_transform(XtrainAll,label)

###------------------------------------------- PCA
#from sklearn.decomposition import PCA
#pca = PCA(n_components=30)
####------------------------------------------- Randomized PCA
#from sklearn.decomposition import RandomizedPCA
#pca = RandomizedPCA(n_components=30, whiten=True)
##
##
#scale = pca.fit(XtrainPos)
#XtrainPos = scale.fit_transform(XtrainPos)
#XtestPos = scale.fit_transform(XtestPos)
#scaleAll = pca.fit(XtrainAll)
#XtrainAll = scaleAll.transform(XtrainAll)

###------------------------------------------- LDA
#from sklearn.lda import LDA
#lda = LDA(n_components=4)
#scale = lda.fit(XtrainPos,YtrainPos)
#XtrainPos = scale.transform(XtrainPos)
#XtestPos = scale.transform(XtestPos)
#scaleAll = lda.fit(XtrainAll,label)
#XtrainAll = scaleAll.transform(XtrainAll)


#--------------------feature Ranking---------------------------------

from sklearn.feature_selection import RFE
## SVM                                     
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
rfe = RFE(clf, 3)
rfe = rfe.fit(XtrainAll, label)
print(rfe.support_)
print(rfe.ranking_)

#ExtraTreesClassifier
#from sklearn.ensemble import ExtraTreesClassifier
#clf = ExtraTreesClassifier()
#clf.fit(XtrainAll, label)
#print(clf.feature_importances_)










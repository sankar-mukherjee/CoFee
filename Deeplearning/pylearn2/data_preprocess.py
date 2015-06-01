"""
Created on Tue Feb 24 16:08:39 2015

@author: mukherjee

data preprocessing and train test valid creation for plearn2 experiments
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing

# read Form data
DATA_FORM_FILE = '../data/all-merged-cat.csv'
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


TEST_FRAC = 0.15
VALID_FRAC = 0.15

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
nSamples = len(feat)

test_set_size = int(TEST_FRAC * len(feat))
valid_set_size = int(VALID_FRAC * len(feat))

Xtest = feat[0:test_set_size,:]
Xvalid = feat[test_set_size:test_set_size+valid_set_size,:]
Xtrain = feat[test_set_size+valid_set_size:,:]

Ytest = label[0:test_set_size]
Yvalid = label[test_set_size:test_set_size+valid_set_size]
Ytrain = label[test_set_size+valid_set_size:]

#----------------------------------------------------------------------------------------------------
#normalization of features
scale = preprocessing.StandardScaler().fit(Xtrain)
Xtrain = scale.transform(Xtrain)
Xtest = scale.transform(Xtest)
Xvalid = scale.transform(Xvalid)

#scale = preprocessing.MinMaxScaler()
#Xtrain = scale.fit_transform(Xtrain)
#Xtest = scale.transform(Xtest)
#Xvalid = scale.transform(Xvalid)
#
#scale = preprocessing.Normalizer().fit(Xtrain)
#Xtrain = scale.transform(Xtrain)
#Xtest = scale.transform(Xtest)
#Xvalid = scale.transform(Xvalid)

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
#scale = lda.fit(Xtrain,Ytrain)
#Xtrain = scale.transform(Xtrain)
#Xtest = scale.transform(Xtest)
#Xvalid = scale.transform(Xvalid)

#-----------saving data as CSV--------------------------------

train =  np.column_stack((Ytrain,Xtrain))
test =  np.column_stack((Ytest,Xtest))
valid =  np.column_stack((Yvalid,Xvalid))

np.savetxt("../data/test.csv", test, delimiter=",",fmt='%g')
np.savetxt("../data/valid.csv", valid, delimiter=",",fmt='%g')
np.savetxt("../data/train.csv", train, delimiter=",",fmt='%g')



"""
Created on Tue Feb 24 16:08:39 2015

@author: mukherjee
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics, cross_validation
from sklearn.ensemble import GradientBoostingClassifier

# read Form data
DATA_FORM_FILE = 'all-merged-cat.csv'
rawdata = pd.read_csv(DATA_FORM_FILE, usecols=np.r_[3,5:12,13:28,81:87,108])

#select features

posfeat = pd.DataFrame.as_matrix(rawdata)[:,np.r_[0:8,10:23]].astype(float)

# select test labels
#Ytest = pd.DataFrame.as_matrix(rawdata)[:,20:26].astype(float)
label = pd.DataFrame.as_matrix(rawdata)[:,29]

#remove bad features as there is no label
scale = np.where(label == 'None')
label = np.delete(label,scale)
posfeat = np.delete(posfeat,scale,0)

# Transforming categorical feature
le = preprocessing.LabelEncoder()
le.fit(label)
list(le.classes_)
label = le.transform(label)

# create traning and test data by partioning
nSamples = len(posfeat)

XtrainPos = posfeat[:.7 * nSamples,:]
YtrainPos = label[:.7 * nSamples]

XtestPos = posfeat[.7 * nSamples:,:]
YtestPos = label[.7 * nSamples:]                

#normalization of features
#scale = preprocessing.StandardScaler().fit(XtrainPos)
#XtrainPos = scale.fit_transform(XtrainPos)
#XtestPos = scale.fit_transform(XtestPos)

#scale = preprocessing.MinMaxScaler()
#XtrainPos = scale.fit_transform(XtrainPos)
#XtestPos = scale.fit_transform(XtestPos)
#
#scale = preprocessing.Normalizer().fit(XtrainPos)
#XtrainPos = scale.fit_transform(XtrainPos)
#XtestPos = scale.fit_transform(XtestPos)

#classification
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.35, 
                                     max_depth=1, random_state=0).fit(XtrainPos, YtrainPos)
print(metrics.classification_report(YtestPos, clf.predict(XtestPos)))

## Crossvalidation 5 times using different split
#scores = cross_validation.cross_val_score(clf_svm, posfeat, label, cv=5, scoring='f1')
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Visualization
#plt.hist(XtrainPos[:,0])
#plt.show()


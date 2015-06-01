"""
Created on Tue Feb 24 16:08:39 2015

@author: mukherjee
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm, metrics, cross_validation
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

# select feature combination
#feat = posfeat
feat = np.column_stack((posfeat,accoufeat,lexfeat))
feat[np.isnan(feat)] = 0
feat[np.isinf(feat)] = 0
# select test labels
#Ytest = pd.DataFrame.as_matrix(rawdata)[:,20:26].astype(float)
label = pd.DataFrame.as_matrix(rawdata)[:,108]

#remove bad features as there is no label
scale = np.where(label == 'None')
label = np.delete(label,scale)
feat = np.delete(feat,scale,0)

# Transforming categorical feature
le = preprocessing.LabelEncoder()
le.fit(label)
list(le.classes_)
label = le.transform(label)

# create traning and test data by partioning
nSamples = len(feat)

XtrainPos = feat[:.7 * nSamples,:]
YtrainPos = label[:.7 * nSamples]

XtestPos = feat[.7 * nSamples:,:]
YtestPos = label[.7 * nSamples:]                

#normalization of features
scale = preprocessing.StandardScaler().fit(XtrainPos)
XtrainPos = scale.fit_transform(XtrainPos)
XtestPos = scale.fit_transform(XtestPos)
# for whole data set
scaleAll = preprocessing.StandardScaler().fit(feat)
XtrainAll = scaleAll.fit_transform(feat)

#scale = preprocessing.MinMaxScaler()
#XtrainPos = scale.fit_transform(XtrainPos)
#XtestPos = scale.fit_transform(XtestPos)
#scaleAll = preprocessing.MinMaxScaler()
#XtrainAll = scaleAll.fit_transform(feat)

#scale = preprocessing.Normalizer().fit(XtrainPos)
#XtrainPos = scale.fit_transform(XtrainPos)
#XtestPos = scale.fit_transform(XtestPos)
#scaleAll = preprocessing.Normalizer().fit(feat)
#XtrainAll = scaleAll.fit_transform(feat)

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
# PCA:
pca = PCA(n_components=2)
XtrainAll = pca.fit(XtrainAll).transform(XtrainAll)
# Maybe some original features where good, too?
selection = SelectKBest(k=1)
# Build estimator from PCA and Univariate selection:
combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
XtrainAll = combined_features.fit(XtrainAll, label).transform(XtrainAll)


# Classify:
clf = svm.SVC()
clf.fit(XtrainAll, label)

# Do grid search over k, n_components and C:

pipeline = Pipeline([("features", combined_features), ("svm", clf)])

param_grid = dict(features__pca__n_components=[1, 2, 3],
                  features__univ_select__k=[1, 2],
                  svm__C=[0.1, 1, 10])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(XtrainAll, label)
print(grid_search.best_estimator_)






#classification
clf = svm.SVC()
#clf_svm = svm.LinearSVC()
clf = clf.fit(XtrainPos, YtrainPos)
print(metrics.classification_report(YtestPos, clf.predict(XtestPos)))

############### Check for overfeat-------------------------------------
train_sample_size, train_scores, test_scores = learning_curve(clf,
                                                              XtrainAll, label, 
                                                              train_sizes=np.arange(10,100,10), cv=3)


# Visualization
#plt.set_title("Learning Curve ("+clsfr_name+")")
plt.xlabel("# Training sample")
plt.ylabel("Accuracy")
plt.grid();
mean_train_scores = np.mean(train_scores, axis=1)
mean_test_scores = np.mean(test_scores, axis=1)
std_train_scores = np.std(train_scores, axis=1)
std_test_scores = np.std(test_scores, axis=1)

gap = np.abs(mean_test_scores - mean_train_scores)
plt.title("Learning curves for %r\n"
             "Best test score: %0.2f - Gap: %0.2f" %
             (clf, mean_test_scores.max(), gap[-1]))
plt.plot(train_sample_size, mean_train_scores, label="Training", color="b")
plt.fill_between(train_sample_size, mean_train_scores - std_train_scores,
                 mean_train_scores + std_train_scores, alpha=0.1, color="b")
plt.plot(train_sample_size, mean_test_scores, label="Cross-validation",
         color="g")
plt.fill_between(train_sample_size, mean_test_scores - std_test_scores,
                 mean_test_scores + std_test_scores, alpha=0.1, color="g")
plt.legend(loc="lower right")


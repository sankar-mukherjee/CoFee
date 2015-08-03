"""
Created on Tue Feb 24 16:08:39 2015

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
posfeat_name = rawdata.columns.values[3:12]

lextypefeat = pd.DataFrame.as_matrix(rawdata)[:,np.r_[12:14]]
lextypefeat_name = rawdata.columns.values[12:14]

lexfeat = pd.DataFrame.as_matrix(rawdata)[:,np.r_[14:29]].astype(float)
lexfeat_name = rawdata.columns.values[14:29]

phonfeat = pd.DataFrame.as_matrix(rawdata)[:,np.r_[29:47]]

accoufeat = pd.DataFrame.as_matrix(rawdata)[:,np.r_[47:81]].astype(float)
accoufeat_name = rawdata.columns.values[47:81]

phonfeat = pd.DataFrame.as_matrix(rawdata)[:,np.r_[29]].astype(float)
lextypefeat = pd.DataFrame.as_matrix(rawdata)[:,np.r_[13]]
lextypefeat_name = rawdata.columns.values[13:14].astype(object)

# feature name
feat_name = np.concatenate((posfeat_name,accoufeat_name,lexfeat_name),axis=0)

# Transforming categorical feature
le = preprocessing.LabelBinarizer()
le.fit(lextypefeat)
list(le.classes_)
lextypefeat = le.transform(lextypefeat)
#----------------------------------------------------------------------------------------------------
# select feature combination
featN = np.column_stack((posfeat,accoufeat))

#featB = np.column_stack((lexfeat,lextypefeat))
featB = lexfeat

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

XtrainPos = feat[:.7 * nSamples,:]
YtrainPos = label[:.7 * nSamples]

XtestPos = feat[.7 * nSamples:,:]
YtestPos = label[.7 * nSamples:]    

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

#--------------------------------------------classification-------------------------------------------
##GradientBoost
#from sklearn.ensemble import GradientBoostingClassifier
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
#                                     max_depth=1, random_state=0)

## SVM                                     
#from sklearn import svm
#clf = svm.SVC()

#from sklearn.multiclass import OneVsOneClassifier
#from sklearn.multiclass import OutputCodeClassifier
#clf = OutputCodeClassifier(svm.SVC())

## RandomForest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(min_samples_leaf=10)

## SGD
#from sklearn.linear_model import SGDClassifier
#clf = SGDClassifier(loss="log", penalty="l2")

# CART
#from sklearn import tree
#clf = tree.DecisionTreeClassifier()
#
### AdaBoostClassifier
#from sklearn.ensemble import AdaBoostClassifier
#clf = AdaBoostClassifier(n_estimators=100)

#  Gaussian Naive Bayes
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

# KNN
#from sklearn import neighbors
##clf = neighbors.KNeighborsClassifier(n_neighbors=10,weights='distance')
#clf = neighbors.KNeighborsClassifier(n_neighbors=10)


##-------------------------------------------------Traning------------------
clf = clf.fit(XtrainPos, YtrainPos)
print(metrics.classification_report(YtestPos, clf.predict(XtestPos)))

##--------------------------Crossvalidation 5 times using different split------------------------------
#from sklearn import cross_validation
#scores = cross_validation.cross_val_score(clf, XtrainAll, label, cv=3, scoring='f1')
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

####---------------------------------Check for overfeat-------------------------------------
train_sample_size, train_scores, test_scores = learning_curve(clf,
                                                              XtrainAll, label, 
                                                              train_sizes=np.arange(0.1,1,0.1), cv=10)

#----------------------------------------Visualization---------------------------------------------
plt.xlabel("# Training sample")
plt.ylabel("Accuracy")
plt.grid();
mean_train_scores = np.mean(train_scores, axis=1)
mean_test_scores = np.mean(test_scores, axis=1)
std_train_scores = np.std(train_scores, axis=1)
std_test_scores = np.std(test_scores, axis=1)

gap = np.abs(mean_test_scores - mean_train_scores)
g = plt.figure(1)
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
g.show()

## confusion matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(YtestPos,clf.predict(XtestPos))
## Show confusion matrix in a separate window
#plt.matshow(cm)
#plt.title('Confusion matrix')
#plt.colorbar()
#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.show()

###############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
f = plt.figure(2,figsize=(18, 18))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, feat_name[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.savefig('feature_importance')
f.show()

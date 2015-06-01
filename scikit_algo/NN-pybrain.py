# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 16:28:49 2015

@author: mukherjee
"""
import numpy as np
import pandas as pd
from math import sqrt
from sklearn import preprocessing, metrics, cross_validation

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError



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
scale = preprocessing.StandardScaler().fit(XtrainPos)
XtrainPos = scale.fit_transform(XtrainPos)
XtestPos = scale.fit_transform(XtestPos)

#scale = preprocessing.MinMaxScaler()
#XtrainPos = scale.fit_transform(XtrainPos)
#XtestPos = scale.fit_transform(XtestPos)
#
scale = preprocessing.Normalizer().fit(XtrainPos)
XtrainPos = scale.fit_transform(XtrainPos)
XtestPos = scale.fit_transform(XtestPos)

# Neural Network 
YtrainPos = YtrainPos.reshape( -1, 1 )  
YtestPos = YtestPos.reshape( -1, 1 )  


input_size = XtrainPos.shape[1]
target_size = YtrainPos.shape[1]
hidden_size = 50   # arbitrarily chosen

#ds = SupervisedDataSet(input_size,target_size )
ds = ClassificationDataSet(21)
ds.setField( 'input', XtrainPos )
ds.setField( 'target', YtrainPos )
ds._convertToOneOfMany(bounds=[0, 1])

net = buildNetwork( input_size, hidden_size, 5, bias = True )
trainer = BackpropTrainer( net, ds )

epochs = 2
print "training for {} epochs...".format( epochs )

for i in range( epochs ):
	mse = trainer.train()
	rmse = sqrt(mse)
	print "training RMSE, epoch {}: {}".format( i + 1, rmse )

#trainer.trainUntilConvergence( verbose = True, validationProportion = 0.15, maxEpochs = 1000, continueEpochs = 10 )

lb = preprocessing.LabelBinarizer()
lb.fit(YtestPos)
list(lb.classes_)
YtestPos = lb.transform(YtestPos)

ds.setField( 'input', XtestPos )
ds.setField( 'target', YtestPos )
x = ds.getField('input')
y = ds.getField('target')

trnresult = percentError( trainer.testOnClassData(),trndata['class'] )
tstresult = percentError( trainer.testOnClassData(dataset=x ), YtestPos.T )

print "epoch: %4d" % trainer.totalepochs, "  train error: %5.2f%%" % trnresult, 
" test error: %5.2f%%" % tstresult








#
#from pybrain.datasets            import ClassificationDataSet
#from pybrain.utilities           import percentError
#from pybrain.tools.shortcuts     import buildNetwork
#from pybrain.supervised.trainers import BackpropTrainer
#from pybrain.structure.modules   import SoftmaxLayer
#
#from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
#from scipy import diag, arange, meshgrid, where
#from numpy.random import multivariate_normal
#
#means = [(-1,0),(2,4),(3,1)]
#cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
#alldata = ClassificationDataSet(2, 1, nb_classes=3)
#for n in xrange(400):
#    for klass in range(3):
#        input = multivariate_normal(means[klass],cov[klass])
#        alldata.addSample(input, [klass])
#        
#tstdata, trndata = alldata.splitWithProportion( 0.25 )
#trndata._convertToOneOfMany()
#tstdata._convertToOneOfMany()

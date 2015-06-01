# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:27:32 2015

@author: sank
"""
import os
import numpy as np

from pylearn2.train import Train
from pylearn2.models import softmax_regression, mlp
from pylearn2.training_algorithms import bgd, sgd
from pylearn2.termination_criteria import MonitorBased
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train_extensions import best_params
from pylearn2.utils import serial
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets import csv_dataset
from theano import function
from theano import tensor as T
import theano

train = csv_dataset.CSVDataset("../data/train.csv",expect_labels=True,expect_headers=False,delimiter=',')
valid = csv_dataset.CSVDataset("../data/valid.csv",expect_labels=True,expect_headers=False,delimiter=',')
test = csv_dataset.CSVDataset("../data/test.csv",expect_labels=True,expect_headers=False,delimiter=',')

# ------------------------------------------Simple ANN

h0 = mlp.Sigmoid(layer_name="h0",dim=73,sparse_init=0)
y0 = mlp.Softmax(n_classes=5,layer_name="y0", irange=0)
layers = [h0, y0]

nn = mlp.MLP(layers,nvis=train.X.shape[1])
algo = sgd.SGD(learning_rate=0.05,batch_size=100,monitoring_dataset=valid,termination_criterion=EpochCounter(100))
algo.setup(nn,train)

save_best = best_params.MonitorBasedSaveBest(channel_name="objective",save_path='best_params.pkl')
while True:
    algo.train(dataset=train)
    nn.monitor.report_epoch()
    nn.monitor()
    save_best.on_monitor(nn,train,algo)
    if not algo.continue_learning(nn):
        break

# SoftPlus with Dropout

h0 = mlp.Softplus(layer_name='h0', dim=60, sparse_init=0)
h1 = mlp.Softplus(layer_name='h1', dim=60, sparse_init=0)
y0 = mlp.Softmax(layer_name='y0', n_classes=5, irange=0)
layers = [h0, h1, y0]

model = mlp.MLP(layers, nvis=train.X.shape[1])

monitoring = dict(valid=valid)
termination = MonitorBased(channel_name="valid_y0_misclass", N=5)
extensions = [best_params.MonitorBasedSaveBest(channel_name="valid_y0_misclass",
save_path="train_best.pkl")]

algorithm = sgd.SGD(0.1, batch_size=100, cost=Dropout(),
                    monitoring_dataset = monitoring, termination_criterion = termination)

print 'Running training'
train_job = Train(train, model, algorithm, extensions=extensions, save_path="train.pkl", save_freq=1)
train_job.main_loop()    


# Rectified Linear with Momentum

from pylearn2.training_algorithms import sgd, learning_rule

h0 = mlp.RectifiedLinear(layer_name='h0', dim=60, sparse_init=0)
y0 = mlp.Softmax(layer_name='y0', n_classes=5, irange=0)
layers = [h0, y0]

model = mlp.MLP(layers, nvis=train.X.shape[1])

# momentum
initial_momentum = 0.5
final_momentum = 0.99
start = 1
saturate = 50
momentum_rule = learning_rule.Momentum(initial_momentum)

monitoring = dict(valid=valid)
termination = MonitorBased(channel_name="valid_y0_misclass", N=10)
extensions = [best_params.MonitorBasedSaveBest(channel_name="valid_y0_misclass",save_path="rect_best.pkl"),
              learning_rule.MomentumAdjustor(final_momentum,start,saturate)]

algorithm = sgd.SGD(0.1, batch_size=100, cost=Dropout(),learning_rule=momentum_rule,
                    monitoring_dataset = monitoring, termination_criterion = termination)

print 'Running training'
train_job = Train(train, model, algorithm, extensions=extensions, save_path="rect.pkl", save_freq=5)
train_job.main_loop()

#------------------------------------------------------------------------------
weight_name = "train_best.pkl"
weight_name = "best_params.pkl"
weight_name = "rect_best.pkl"
# Calculate accuracy on test set
nn = serial.load(weight_name)
inputs = test.X.astype("float32")
yhat = nn.fprop(theano.shared(inputs,name='inputs')).eval()
count = 0.
for i in range(test.X.shape[0]):
    if np.argmax(test.y[i])==np.argmax(yhat[i]):
        count += 1.

print "accuracy = ", count / test.X.shape[0]

import sys
sys.path.append("/home/sank/Desktop/analysis/pylearn2/")
import plot_monitor
plot_monitor.plot_monitor(model_paths=[weight_name], options_out=None, show_codes=["T"])
plot_monitor.plot_monitor(model_paths=[weight_name], options_out=None, show_codes=["BA"])



# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:44:02 2015

@author: sank
"""
train = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.csv_dataset.CSVDataset {
        path: '../data/train.csv',
        one_hot: 1
    },
    model: !obj:pylearn2.models.mlp.MLP {
        layers: [
                 !obj:pylearn2.models.mlp.Sigmoid {
                     layer_name: 'h0',
                     dim: 5,
                     sparse_init: 1,                     
                 },

                 !obj:pylearn2.models.mlp.Softmax {
                     layer_name: 'y',
                     n_classes: 5,
                     irange: 0.
                 }
                ],
        nvis: 4,
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 10000,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        updates_per_batch: 10,
        monitoring_dataset:
            {
                'train' : *train,
                'valid' : !obj:pylearn2.datasets.csv_dataset.CSVDataset {
                        path: '../data/valid.csv',
                        one_hot: 1
                },
                'test'  : !obj:pylearn2.datasets.csv_dataset.CSVDataset {
                        path: '../data/test.csv',
                        one_hot: 1
                }
            },
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.MonitorBased {
                    channel_name: "valid_y_misclass"
                },
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 10000
                }
            ]
        }
    },
    extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
             channel_name: 'valid_y_misclass',
             save_path: "mlp_best.pkl"
        },
    ]
}
"""
from pylearn2.config import yaml_parse
train = yaml_parse.load(train)
train.main_loop()

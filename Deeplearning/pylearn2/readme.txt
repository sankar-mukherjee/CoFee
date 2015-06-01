pylearn2
-------------------------------
type set cast float64 to int64 error


in csv_dataset.py on system python dist-packeges


if self.task == 'regression':
        super(CSVDataset, self).__init__(X=X, y=y)
    else:
        super(CSVDataset, self).__init__(X=X, y=y
                                         y_labels=np.max(y) + 1)
change to

 if self.task == 'regression':
        super(CSVDataset, self).__init__(X=X, y=y)
    else:
        super(CSVDataset, self).__init__(X=X, y=y.astype(int),
                                         y_labels=np.max(y) + 1)

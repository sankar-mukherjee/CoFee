

import numpy as np
from sklearn.learning_curve import learning_curve

def plot_lurning_curve(ax, clsfr, clsfr_name, X_train, y_train, sizes):
    ax.set_title("Learning Curve ("+clsfr_name+")")
    ax.set_xlabel("# Training sample")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1])
    ax.set_yticks(np.arange(0,1,0.1))
    ax.grid()

    train_sample_size, train_scores, test_scores = learning_curve(
        clsfr, X_train, y_train, cv=5, n_jobs=3, train_sizes=sizes)
    mean_train_scores = np.mean(train_scores, axis=1)
    mean_test_scores = np.mean(test_scores, axis=1)
    std_train_scores = np.std(train_scores, axis=1)
    std_test_scores = np.std(test_scores, axis=1)
    ax.plot(train_sample_size, mean_train_scores, label="Training accuracy", color="b")
    ax.fill_between(train_sample_size, mean_train_scores - std_train_scores,
                     mean_train_scores + std_train_scores, alpha=0.1, color="b")
    ax.plot(train_sample_size, mean_test_scores, label="Cross-validation accuracy",
             color="g")
    ax.fill_between(train_sample_size, mean_test_scores - std_test_scores,
                     mean_test_scores + std_test_scores, alpha=0.1, color="g")
    ax.legend(loc="lower right")
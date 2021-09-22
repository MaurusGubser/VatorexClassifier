import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve

from data_reading_writing import read_data_and_labels
from data_handling import split_and_sample_data


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, train_sizes=np.linspace(.1, 1.0, 5)):
    _, axes = plt.subplots(1, 3, figsize=(20, 15))
    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y, cv=cv, n_jobs=-1,
                                                                          train_sizes=train_sizes, return_times=True,
                                                                          verbose=2)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    plt.show()
    return None


def plot_learning_curve_model(folder_path, data_params, model, model_name, undersampling_rate, oversampling_rate):
    data, labels, paths = read_data_and_labels(folder_path, data_params)
    data, labels, _ = split_and_sample_data(data=data,
                                            labels=labels,
                                            paths_imgs=paths,
                                            test_size=None,
                                            undersampling_rate=undersampling_rate,
                                            oversampling_rate=oversampling_rate)
    cv = 10
    plot_learning_curve(model, model_name, data, labels, ylim=None, cv=cv)
    return None


def compute_precisionrecall_curve(clf, X_train, X_test, y_train, y_test):
    plot_train = plot_precision_recall_curve(clf, X_train, y_train)
    plot_train.ax_.set_title('Precision-recall training set')
    plot_test = plot_precision_recall_curve(clf, X_test, y_test)
    plot_test.ax_.set_title('Precision-recall test set')
    plt.show()
    return None


def compute_roc_curve(clf, X_train, X_test, y_train, y_test):
    plot_train = plot_roc_curve(clf, X_train, y_train)
    plot_train.ax_.set_title('ROC training set')
    plot_test = plot_roc_curve(clf, X_test, y_test)
    plot_test.ax_.set_title('ROC test set')
    plt.show()
    return None


def plot_roc_precrcll_curves(clf, dir_data, data_params, test_size, undersampling_rate, oversampling_rate):
    data, labels, paths_imgs = read_data_and_labels(dir_data, data_params)
    X_train, X_test, y_train, y_test, _, _ = split_and_sample_data(data=data,
                                                                   labels=labels,
                                                                   paths_imgs=paths_imgs,
                                                                   test_size=test_size,
                                                                   undersampling_rate=undersampling_rate,
                                                                   oversampling_rate=oversampling_rate)
    clf.fit(X_train, y_train)
    compute_precisionrecall_curve(clf, X_train, X_test, y_train, y_test)
    compute_roc_curve(clf, X_train, X_test, y_train, y_test)
    return None

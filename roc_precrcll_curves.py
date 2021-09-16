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


def plot_scores(clf, dir_data, data_params, test_size, undersampling_rate, oversampling_rate):
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


# ----- data parameters -----
read_image = False  # True or False
read_hist = 'context'  # must be 'candidate', 'context' or False
with_image = None  # must be None or a scalar, which defines downsize factor; use image
with_binary_patterns = False  # use local binary patterns of image
histogram_params = None  # (3, 16)  # must be None or a tuple of two integers, which describes (nb_divisions, nb_bins)
nb_segments = None  # must be None or a integer; segment image using k-means in color space
threshold_low_var = None  # must be None or a float in [0.0, 1.0], which defines threshold for minimal variance
nb_components_pca = None  # must be None or a integer, which defines number of components
batch_size_pca = None  # must be an integer, should be >= nb_features (ideally larger) and <= nb_images
hist_hsl = True
hist_h = True
hist_s = True
hist_l = True
quadratic_features = False  # use basis 1, x_i, x_i**2, no mixed terms
with_mean = False  # data gets shifted such that mean is 0.0
with_std = False  # data gets scaled such that std is 1.0

data_parameters = OrderedDict([('read_image', read_image), ('read_hist', read_hist), ('with_image', with_image),
                               ('with_binary_patterns', with_binary_patterns), ('histogram_params', histogram_params),
                               ('nb_segments', nb_segments), ('threshold_low_var', threshold_low_var),
                               ('nb_components_pca', nb_components_pca), ('batch_size_pca', batch_size_pca),
                               ('hist_hsl', hist_hsl), ('hist_h', hist_h), ('hist_s', hist_s), ('hist_l', hist_l),
                               ('quadratic_features', quadratic_features), ('with_mean', with_mean),
                               ('with_std', with_std)])
test_size = 0.10  # fraction of test set
undersampling_rate = None  # desired percentage of trues in training data set
oversampling_rate = None

# ----------- execute functions ----------------------
clf = RandomForestClassifier(class_weight='balanced')
dir_data = 'Candidate_Images/Mite4_relabelledtol05_local/'

plot_scores(clf=clf,
            dir_data=dir_data,
            data_params=data_parameters,
            test_size=test_size,
            undersampling_rate=undersampling_rate,
            oversampling_rate=oversampling_rate)

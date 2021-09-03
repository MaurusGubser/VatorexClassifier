import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve

from data_reading_writing import read_data_and_labels
from data_handling import split_and_sample_data


def compute_precrcll_curve(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    plot_train = plot_precision_recall_curve(clf, X_train, y_train)
    plot_train.ax_.set_title('Precision-recall training set')
    plot_test = plot_precision_recall_curve(clf, X_test, y_test)
    plot_test.ax_.set_title('Precision-recall test set')
    plt.show()
    return None


def compute_roc_curve(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    plot_train = plot_roc_curve(clf, X_train, y_train)
    plot_train.ax_.set_title('ROC training set')
    plot_test = plot_roc_curve(clf, X_test, y_test)
    plot_test.ax_.set_title('ROC test set')
    plt.show()
    return None


def plot_scores(clf, dir_data, data_params, test_size, percentage_true):
    data, labels, paths_imgs = read_data_and_labels(dir_data, data_params)
    X_train, y_train, _, X_test, y_test, _ = split_and_sample_data(data, labels, paths_imgs, test_size, percentage_true)
    compute_precrcll_curve(clf, X_train, X_test, y_train, y_test)
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
percentage_true = 0.10  # desired percentage of trues in training data set

# ----------- execute functions ----------------------
clf = LogisticRegression()
dir_data = '/home/maurus/PyCharm_Projects/Vatorex_Classifier/Candidate_Images/Mite4_relabelledtol05/'

plot_scores(clf=clf, dir_data=dir_data)

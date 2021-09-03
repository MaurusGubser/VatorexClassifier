import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve, precision_recall_curve, roc_curve

from data_reading_writing import read_data_and_labels
from data_handling import downsize_false_candidates


def compute_precrcll_curve(clf, data_dir, data_params, test_size, percentage_true):
    data, labels, paths_imgs = read_data_and_labels(data_dir, data_params)
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(data,
                                                                                 labels,
                                                                                 paths_imgs,
                                                                                 test_size=test_size,
                                                                                 random_state=42,
                                                                                 stratify=labels)
    X_train, y_train, paths_train = downsize_false_candidates(X_train, y_train, paths_train, percentage_true)
    clf.fit(X_train, y_train)

    probs_train = clf.predict_proba(X_train)
    prec_train, rcll_train, thres_train = precision_recall_curve(y_train, probs_train[:, 1])
    print('Precision training data:\n', prec_train)
    print('Recall training data:\n', rcll_train)
    dsp = plot_precision_recall_curve(clf, X_train, y_train)
    dsp.ax_.set_title('Precision-recall training set')
    plt.show()

    probs_test = clf.predict_proba(X_test)
    prec_test, rcll_test, thres_test = precision_recall_curve(y_test, probs_test[:, 1])
    print('Precision test data:\n', prec_test)
    print('Recall test data:\n', rcll_test)
    dsp = plot_precision_recall_curve(clf, X_test, y_test)
    dsp.ax_.set_title('Precision-recall test set')
    plt.show()

    return None


def compute_roc_curve(clf, data_dir, data_params, test_size, percentage_true):
    data, labels, paths_imgs = read_data_and_labels(data_dir, data_params)
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(data,
                                                                                 labels,
                                                                                 paths_imgs,
                                                                                 test_size=test_size,
                                                                                 random_state=42,
                                                                                 stratify=labels)
    X_train, y_train, paths_train = downsize_false_candidates(X_train, y_train, paths_train, percentage_true)
    clf.fit(X_train, y_train)

    probs_train = clf.predict_proba(X_train)
    fpr_train, tpr_train, thres_train = roc_curve(y_train, probs_train[:, 1])
    print('False positive rate train:\n', fpr_train)
    print('True positive rate train:\n', tpr_train)
    dsp = plot_roc_curve(clf, X_train, y_train)
    dsp.ax_.set_title('ROC training set')
    plt.show()

    probs_test = clf.predict_proba(X_test)
    fpr_test, tpr_test, thres_test = roc_curve(y_test, probs_test[:, 1])
    print('False positive rate test:\n', fpr_test)
    print('True positive rate test:\n', tpr_test)
    dsp = plot_roc_curve(clf, X_test, y_test)
    dsp.ax_.set_title('ROC test set')
    plt.show()

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
path_dir = '/home/maurus/PyCharm_Projects/Vatorex_Classifier/Candidate_Images/Mite4_relabelledtol05/'

compute_roc_curve(clf=clf,
                  data_dir=path_dir,
                  data_params=data_parameters,
                  test_size=test_size,
                  percentage_true=percentage_true)

compute_precrcll_curve(clf=clf,
                       data_dir=path_dir,
                       data_params=data_parameters,
                       test_size=test_size,
                       percentage_true=percentage_true)

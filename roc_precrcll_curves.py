import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.svm import LinearSVC
from collections import OrderedDict

from data_reading_writing import read_data_and_labels_from_path
from data_handling import split_and_sample_data


def plot_learning_curve(estimator: object, title: str, X: np.ndarray, y: np.ndarray, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)) -> None:
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


def plot_learning_curve_model(folder_path: str, data_params: dict, model: object, model_name: str) -> None:
    data, labels, paths = read_data_and_labels_from_path(folder_path, data_params)
    # X_train, X_test, y_train, y_test, paths_train, paths_test
    data, _, labels, _, _, _ = split_and_sample_data(data=data,
                                                     labels=labels,
                                                     paths_imgs=paths,
                                                     test_size=None)
    cv = 10
    plot_learning_curve(model, model_name, data, labels, ylim=None, cv=cv)
    return None


def compute_metric_curve(metric_type: [RocCurveDisplay, PrecisionRecallDisplay], name: str, clf: object,
                         X_test: np.ndarray, y_test: np.ndarray):
    try:
        plot_test = metric_type.from_estimator(clf, X_test, y_test)
        plot_test.ax_.set_title('{} test set'.format(name))
    except ValueError:
        plot_test = metric_type.from_predictions(y_true=y_test, y_pred=clf.predict(X_test))
        plot_test.ax_.set_title('{} test set'.format(name))
    return plot_test


def plot_roc_precrcll_curves(clf: object, dir_data: str, data_params: dict, test_size: float) -> None:
    data, labels, paths_imgs = read_data_and_labels_from_path(dir_data, data_params)
    X_train, X_test, y_train, y_test, _, _ = split_and_sample_data(data=data,
                                                                   labels=labels,
                                                                   paths_imgs=paths_imgs,
                                                                   test_size=test_size)
    clf.fit(X_train, y_train)
    figs = []
    metrics = {'ROC': RocCurveDisplay, 'Precision-Recall': PrecisionRecallDisplay}
    for name, metric in metrics.items():
        figs.append(compute_metric_curve(metric, name, clf, X_test, y_test))
        plt.show()
    return figs

"""
# ------------------------ remove -------------------------
read_image = False  # True or False
read_hist = 'context'  # must be 'candidate', 'context' or None
with_image = None  # must be None or a scalar, which defines downsize factor; use image
with_binary_patterns = False  # use local binary patterns of image
histogram_params = None  # (3, 16)  # must be None or a tuple of two integers, which describes (nb_divisions, nb_bins)
nb_segments = None  # must be None or an integer; segment image using k-means in color space
threshold_low_var = None  # must be None or a float in [0.0, 1.0], which defines threshold for minimal variance
nb_components_pca = None  # must be None or an integer, which defines number of components
batch_size_pca = None  # must be an integer, should be >= nb_features (ideally larger) and <= nb_images
hist_hsl = True
hist_h = True
hist_s = True
hist_l = True
quadratic_features = False  # use basis 1, x_i, x_i**2, no mixed terms
with_mean = False  # data gets shifted such that mean is 0.0
with_std = False  # data gets scaled such that std is 1.0
with_false1 = False  # use false1 labelled data

path = 'Candidate_Images/Series_matching05_mindist015_test'
params = OrderedDict([('read_image', read_image), ('read_hist', read_hist), ('with_image', with_image),
                      ('with_binary_patterns', with_binary_patterns), ('histogram_params', histogram_params),
                      ('nb_segments', nb_segments), ('threshold_low_var', threshold_low_var),
                      ('nb_components_pca', nb_components_pca), ('batch_size_pca', batch_size_pca),
                      ('hist_hsl', hist_hsl), ('hist_h', hist_h), ('hist_s', hist_s), ('hist_l', hist_l),
                      ('quadratic_features', quadratic_features), ('with_mean', with_mean),
                      ('with_std', with_std), ('with_false1', with_false1)])
mdl = LinearSVC()
name = 'LinearSVC'

plot_learning_curve_model(folder_path=path, data_params=params, model=mdl, model_name=name)
"""
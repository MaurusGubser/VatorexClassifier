import json
import os
import pandas as pd
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve, learning_curve, train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

from data_handling import downsize_false_candidates
from data_reading_writing import read_data_and_labels
from model_train_test import get_name_index


def compute_cv_scores(model_type, data, labels, cv_params, score_param):
    model_parameter = cv_params['model_parameter']
    parameter_range = cv_params['parameter_range']
    k = cv_params['nb_split_cv']
    train_scores, test_scores = validation_curve(estimator=model_type, X=data, y=labels, param_name=model_parameter,
                                                 param_range=parameter_range, cv=k, scoring=score_param,
                                                 n_jobs=-1, verbose=2)
    print('Train scores {}: {}'.format(score_param, train_scores))
    print('Test scores {}: {}'.format(score_param, test_scores))
    return train_scores, test_scores


def plot_validation_curve(train_scores, test_scores, cv_params):
    if not os.path.exists('CV_Plots'):
        os.mkdir('CV_Plots')
    parameter_range = cv_params['parameter_range']
    export_name = cv_params['model_name'] + '_' + cv_params['model_parameter']
    fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(18, 10))
    i = 0
    for key in train_scores.keys():
        train_scores_mean = np.mean(train_scores[key], axis=1)
        train_scores_std = np.std(train_scores[key], axis=1)
        test_scores_mean = np.mean(test_scores[key], axis=1)
        test_scores_std = np.std(test_scores[key], axis=1)
        axs[i].grid()
        if cv_params['semilog']:
            axs[i].semilogx(parameter_range, train_scores_mean, color='blue', label='Training')
            axs[i].semilogx(parameter_range, test_scores_mean, color='red', label='Test')
        else:
            axs[i].plot(parameter_range, train_scores_mean, color='blue', label='Training')
            axs[i].plot(parameter_range, test_scores_mean, color='red', label='Test')
        axs[i].fill_between(parameter_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                            color='blue', alpha=0.2)
        axs[i].fill_between(parameter_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                            color='red', alpha=0.2)
        axs[i].set_ylim(0.0, 1.1)
        axs[i].set_xlabel(cv_params['model_parameter'])
        axs[i].set_ylabel(key)
        axs[i].set_title('{}-fold cross validation'.format(cv_params['nb_split_cv']))
        axs[i].legend()
        i += 1
    plt.tight_layout()
    plt.savefig('CV_Plots/' + export_name)
    plt.show()

    return None


def cross_validate_model(model, folder_path, data_params, cv_params):
    data, labels = read_data_and_labels(folder_path, data_params)
    data, labels = downsize_false_candidates(data, labels, data_params['percentage_true'])
    indices = np.arange(labels.shape[0])
    np.random.shuffle(indices)
    data, labels = data[indices], labels[indices]
    train_scores = OrderedDict({})
    test_scores = OrderedDict({})
    for score_param in ['recall', 'precision', 'f1']:
        train_scores[score_param], test_scores[score_param] = compute_cv_scores(model, data, labels, cv_params,
                                                                                score_param)
    plot_validation_curve(train_scores, test_scores, cv_params)

    return None


def export_stats_gs(model_name, gs_dataframe):
    if not os.path.exists('GridSearch_Statistics'):
        os.mkdir('GridSearch_Statistics')
    model_nb = get_name_index(model_name, 'GridSearch_Statistics/', 'csv')
    rel_file_path = 'GridSearch_Statistics/' + model_name + '_' + str(model_nb) + '.csv'
    gs_dataframe.to_csv(rel_file_path)
    print("GridSearch statistics saved in", rel_file_path)
    return None


def grid_search_model(model, folder_path, data_params, grid_search_params):
    data, labels = read_data_and_labels(folder_path, data_params)
    data, labels = downsize_false_candidates(data, labels, data_params['percentage_true'])
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.05, shuffle=True, random_state=42)

    clf = GridSearchCV(model, grid_search_params['parameters_grid'], grid_search_params['scoring_parameters'],
                       n_jobs=-1, refit='recall', cv=grid_search_params['nb_split_cv'], verbose=2)
    clf.fit(X_train, y_train)
    print('Best estimator:', clf.best_estimator_)
    gs_df = pd.DataFrame.from_dict(clf.cv_results_)
    gs_df = gs_df[gs_df['rank_test_recall'] <= 10]
    export_stats_gs(grid_search_params['model_name'], gs_df)

    return None


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


def plot_learning_curve_model(folder_path, data_params, model, model_name):
    data, labels = read_data_and_labels(folder_path, data_params)
    data, labels = downsize_false_candidates(data, labels, data_params['percentage_true'])

    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    cv = 5
    plot_learning_curve(model, model_name, data, labels, ylim=None, cv=cv)
    return None

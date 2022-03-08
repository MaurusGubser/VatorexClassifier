import os
import re
import pandas as pd
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from typing import Union, List

from data_handling import split_and_sample_data, compute_prior_weight
from data_reading_writing import read_data_and_labels
from model_train_test import get_name_index, evaluate_model, export_evaluation_images_model, export_model, \
    export_model_evaluation_stats_json


def compute_cv_scores(model_type: object, data: np.ndarray, labels: np.ndarray, cv_params: dict, score_param: str,
                      weights_dict: dict) -> (np.ndarray, np.ndarray):
    model_parameter = cv_params['model_parameter']
    parameter_range = cv_params['parameter_range']
    k = cv_params['nb_split_cv']
    train_scores, test_scores = validation_curve(estimator=model_type, X=data, y=labels, param_name=model_parameter,
                                                 param_range=parameter_range, cv=k, scoring=score_param,
                                                 n_jobs=-1, verbose=1, fit_params=weights_dict)
    return train_scores, test_scores


def compute_f1(train_scores, test_scores):
    train_f1 = 2 * train_scores['recall'] * train_scores['precision'] / (train_scores['recall'] + train_scores['precision'])
    test_f1 = 2 * test_scores['recall'] * test_scores['precision'] / (test_scores['recall'] + test_scores['precision'])
    return train_f1, test_f1


def plot_validation_curve(train_scores: dict, test_scores: dict, cv_params: dict) -> None:
    if not os.path.exists('CV_Plots'):
        os.mkdir('CV_Plots')
    parameter_range = cv_params['parameter_range']
    model_nb = get_name_index(cv_params['model_name'], 'CV_Plots/', 'pdf')
    export_name = cv_params['model_name'] + '_' + cv_params['model_parameter'] + '_' + str(model_nb)
    nb_rows = len(train_scores.keys())
    fig, axs = plt.subplots(ncols=1, nrows=nb_rows, figsize=(18, 10))
    for i, key in enumerate(train_scores.keys()):
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
    plt.tight_layout()
    plt.savefig('CV_Plots/' + export_name)
    plt.show()
    return None


def cross_validate_model(model: object, folder_path: str, data_params: dict, cv_params: dict,
                         undersampling_rate: Union[None, float], oversampling_rate: Union[None, float],
                         use_weights: Union[None, str, List[float]]) -> None:
    test_size = None
    data, labels, paths_imgs = read_data_and_labels(folder_path, data_params)
    X_train, _, y_train, _, _, _ = split_and_sample_data(data=data,
                                                         labels=labels,
                                                         paths_imgs=paths_imgs,
                                                         test_size=test_size,
                                                         undersampling_rate=undersampling_rate,
                                                         oversampling_rate=oversampling_rate)

    if use_weights == 'balanced' or use_weights is None:
        weights_dict = {'sample_weight': None}
    else:
        nb_samples = labels.size
        nb_pos = np.sum(labels)
        nb_neg = nb_samples - nb_pos
        weights = np.zeros(nb_samples)
        weight_0, weight_1 = data_params['use_weights']
        weights[labels == 0] = weight_0 * nb_samples / (2 * nb_neg)
        weights[labels == 1] = weight_0 * nb_samples / (2 * nb_pos)
        weights_dict = {'sample_weight': weights}

    train_scores = OrderedDict({})
    test_scores = OrderedDict({})

    for score_param in ['recall', 'precision']:
        train_scores[score_param], test_scores[score_param] = compute_cv_scores(model_type=model,
                                                                                data=X_train,
                                                                                labels=y_train,
                                                                                cv_params=cv_params,
                                                                                score_param=score_param,
                                                                                weights_dict=weights_dict)
    train_scores['f1'], test_scores['f1'] = compute_f1(train_scores, test_scores)
    plot_validation_curve(train_scores, test_scores, cv_params)
    return None


def export_stats_gs(export_name: str, gs_dataframe: pd.DataFrame) -> None:
    if not os.path.exists('GridSearch_Statistics'):
        os.mkdir('GridSearch_Statistics')
    rel_file_path = 'GridSearch_Statistics/' + export_name + '.csv'
    gs_dataframe.to_csv(rel_file_path)
    print("GridSearch statistics saved in", rel_file_path)
    return None


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    pattern_datasplits = r'split[0-9]{1,2}[_]'
    pattern_time = r'[_]time'
    pattern_rank = r'rank[_]'
    column_names = df.columns
    for name in column_names:
        if re.search(pattern_datasplits, name) or re.search(pattern_time, name) or re.search(pattern_rank, name):
            df = df.drop(name, axis=1)
    return df


def grid_search_model(model: object, folder_path: str, data_params: dict, grid_search_params: dict, test_size: float,
                      undersampling_rate: Union[None, float], oversampling_rate: Union[None, float],
                      use_weights: Union[None, str, List[float]], reweight_posterior: bool) -> None:
    data, labels, paths_imgs = read_data_and_labels(folder_path, data_params)
    X_train, X_test, y_train, y_test, paths_train, paths_test = split_and_sample_data(data=data,
                                                                                      labels=labels,
                                                                                      paths_imgs=paths_imgs,
                                                                                      test_size=test_size,
                                                                                      undersampling_rate=undersampling_rate,
                                                                                      oversampling_rate=oversampling_rate)
    if reweight_posterior:
        prior_mite, prior_no_mite = compute_prior_weight(np.array(labels), y_train)
    else:
        prior_mite, prior_no_mite = None, None

    if use_weights == 'balanced' or use_weights is None:
        weights_dict = {'sample_weight': None}
    else:
        nb_samples = y_train.size
        nb_pos = np.sum(y_train)
        nb_neg = nb_samples - nb_pos
        weights = np.zeros(nb_samples)
        weight_0, weight_1 = data_params['use_weights']
        weights[labels == 0] = weight_0 * nb_samples / (2 * nb_neg)
        weights[labels == 1] = weight_0 * nb_samples / (2 * nb_pos)
        weights_dict = {'sample_weight': weights}

    clf = GridSearchCV(estimator=model,
                       param_grid=grid_search_params['parameters_grid'],
                       scoring=grid_search_params['scoring_parameters'],
                       n_jobs=-1,
                       refit=grid_search_params['refit_param'],
                       cv=grid_search_params['nb_split_cv'],
                       verbose=2,
                       return_train_score=True)
    clf.fit(X_train, y_train)
    gs_df = pd.DataFrame.from_dict(clf.cv_results_)
    gs_df = clean_df(gs_df)

    model_nb = get_name_index(grid_search_params['model_name'], 'GridSearch_Statistics/', 'csv')
    export_name = grid_search_params['model_name'] + '_' + str(model_nb)
    export_stats_gs(export_name, gs_df)
    _, misclassified_train, true_pos_train = evaluate_model(clf, X_train, y_train, paths_train, prior_mite,
                                                            prior_no_mite)
    stats_test, misclassified_test, true_pos_test = evaluate_model(clf, X_test, y_test, paths_test, prior_mite,
                                                                   prior_no_mite)
    """
    # exclude for now
    if misclassified_train is not None:
        export_evaluation_images_model(misclassified_train, true_pos_train, export_name, 'Train')
    """
    export_evaluation_images_model(misclassified_test, true_pos_test, export_name, 'Test')
    export_model_evaluation_stats_json(stats_test, export_name)
    print('Best estimator:', clf.best_estimator_)
    export_model(clf.best_estimator_, export_name)
    print('Testing score:', clf.score(X_test, y_test))
    plot_confusion_matrix(clf, X_test, y_test)
    plt.show()
    return None

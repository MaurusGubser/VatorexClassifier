import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve

from data_reading_writing import read_data_and_labels
from data_handling import downsize_false_candidates


def compute_cv_scores(model_type, data, labels, cv_params):
    model_parameter = cv_params['model_parameter']
    parameter_range = cv_params['parameter_range']
    k = cv_params['nb_split_cv']
    scoring_parameter = cv_params['scoring_parameter']
    train_scores, test_scores = validation_curve(estimator=model_type, X=data, y=labels, param_name=model_parameter,
                                                 param_range=parameter_range, cv=k, scoring=scoring_parameter)
    print('Train scores: {}'.format(train_scores))
    print('Test scores: {}'.format(test_scores))
    return train_scores, test_scores


def plot_validation_curve(train_scores, test_scores, cv_params):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    parameter_range = cv_params['parameter_range']

    fig = plt.figure()
    ax = fig.gca()
    if cv_params['semilog']:
        ax.semilogx(parameter_range, train_scores_mean, color='blue', label='Training')
        ax.semilogx(parameter_range, test_scores_mean, color='red', label='Test')
    else:
        ax.plot(parameter_range, train_scores_mean, color='blue', label='Training')
        ax.plot(parameter_range, test_scores_mean, color='red', label='Test')
    ax.fill_between(parameter_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                    color='blue', alpha=0.2)
    ax.fill_between(parameter_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                    color='red', alpha=0.2)
    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel(cv_params['model_parameter'])
    ax.set_ylabel(cv_params['scoring_parameter'])
    ax.set_title('{}-fold cross validation'.format(cv_params['nb_split_cv']))
    ax.legend()
    plt.show()

    return None


def cross_validate_model(model, folder_path, data_params, cv_params):
    data, labels = read_data_and_labels(folder_path, data_params)
    data, labels = downsize_false_candidates(data, labels, data_params['percentage_true'])
    indices = np.arange(labels.shape[0])
    np.random.shuffle(indices)
    data, labels = data[indices], labels[indices]

    train_scores, test_scores = compute_cv_scores(model, data, labels, cv_params)
    plot_validation_curve(train_scores, test_scores, cv_params)
    return None

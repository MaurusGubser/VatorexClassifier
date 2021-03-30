import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve, learning_curve, train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

from data_handling import downsize_false_candidates
from data_reading_writing import read_data_and_labels


def compute_cv_scores(model_type, data, labels, cv_params):
    model_parameter = cv_params['model_parameter']
    parameter_range = cv_params['parameter_range']
    k = cv_params['nb_split_cv']
    scoring_parameter = cv_params['scoring_parameter']
    train_scores, test_scores = validation_curve(estimator=model_type, X=data, y=labels, param_name=model_parameter,
                                                 param_range=parameter_range, cv=k, scoring=scoring_parameter,
                                                 n_jobs=-1, verbose=2)
    print('Train scores: {}'.format(train_scores))
    print('Test scores: {}'.format(test_scores))
    return train_scores, test_scores


def plot_validation_curve(train_scores, test_scores, cv_params):
    if not os.path.exists('CV_Plots'):
        os.mkdir('CV_Plots')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    parameter_range = cv_params['parameter_range']
    export_name = cv_params['model_name'] + '_' + cv_params['model_parameter'] + '_' + cv_params['scoring_parameter']
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
    plt.savefig('CV_Plots/' + export_name)
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


def grid_search_model(model, folder_path, data_params, grid_search_params):
    data, labels = read_data_and_labels(folder_path, data_params)
    data, labels = downsize_false_candidates(data, labels, data_params['percentage_true'])
    # indices = np.arange(labels.shape[0])
    # np.random.shuffle(indices)
    # data, labels = data[indices], labels[indices]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.05, shuffle=True, random_state=42)

    clf = GridSearchCV(model, grid_search_params['model_params'], grid_search_params['scoring_parameter'], n_jobs=-1,
                       verbose=2)
    clf.fit(X_train, y_train)
    print('Classifier stats:', clf.cv_results_)
    y_pred = clf.predict(X_test)
    print('Conf matrix predict:', confusion_matrix(y_test, y_pred))

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

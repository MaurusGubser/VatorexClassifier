import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, cross_val_score, cross_validate, \
    validation_curve
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, \
    HistGradientBoostingClassifier

from data_reading_writing import load_data_and_labels, concatenate_data
from data_handling import downsize_false_candidates

read_image = False
read_hist = True
path_data = 'Preprocessed_Data/Mite4_Dataset_Cleaned_False_True_False_False_(3, 32)_10_None_100_1000_True_True_True_True_0.05_False_False.npz'
#path_data = 'Preprocessed_Data/Small_Dataset_True_False_False_False_(1, 64)_5_None_100_1000_False_True_True_True_False_False.npz'
data_images, data_histograms, labels = load_data_and_labels(path_data)
data = concatenate_data(data_images, data_histograms, read_image, read_hist)
data, labels = downsize_false_candidates(data, labels, 0.3)
indices = np.arange(labels.shape[0])
np.random.shuffle(indices)
data, labels = data[indices], labels[indices]

nl_svm_models = [SVC(C=0.1, class_weight='balanced'),
                 SVC(C=1.0, class_weight='balanced'),
                 SVC(C=5.0, class_weight='balanced')]

histogram_boost_models = [HistGradientBoostingClassifier(max_iter=100),
                          HistGradientBoostingClassifier(max_iter=100, l2_regularization=0.1),
                          HistGradientBoostingClassifier(max_iter=100, l2_regularization=1.0),
                          HistGradientBoostingClassifier(max_iter=100, l2_regularization=5.0)]

models = histogram_boost_models
score = ['balanced_accuracy', 'recall', 'precision', 'f1']


def cross_validation_model(model, data, labels, score):
    cv = 10
    return cross_validate(model, X=data, y=labels, scoring=score, cv=cv, return_train_score=True)


def plot_validation_curve(model_type, data, labels, model_parameter, parameter_range):
    train_scores, test_scores = validation_curve(estimator=model_type, X=data, y=labels, param_name=model_parameter,
                                                 param_range=parameter_range, cv=5, scoring='f1')
    print('Training scores f1: {}'.format(train_scores))
    print('Test scores f1: {}'.format(test_scores))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    print('Training std: {}'.format(train_scores_std))
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print('Test std: {}'.format(test_scores_std))
    fig = plt.figure()
    ax = fig.gca()
    ax.semilogx(parameter_range, train_scores_mean, color='blue', label='Training')
    ax.fill_between(parameter_range, train_scores_mean - train_scores_std, train_scores_mean+train_scores_std, color='blue', alpha=0.2)
    ax.semilogx(parameter_range, test_scores_mean, color='red', label='Test')
    ax.fill_between(parameter_range, test_scores_mean - test_scores_std, test_scores_mean+test_scores_std, color='red', alpha=0.2)
    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Score value')
    ax.set_title(model_parameter)
    ax.legend()
    plt.show()
    plt.savefig()
    return None


"""
cv_eval = []
for model in models:
    cv_eval_model = cross_validation_model(model, data, labels, score)
    cv_eval.append(cv_eval_model)
    print(cv_eval_model)
"""

model = HistGradientBoostingClassifier(max_iter=100)
model_parameter = 'l2_regularization'
parameter_range = np.logspace(-4, 1, 10)
plot_validation_curve(model, data, labels, model_parameter, parameter_range)

import os
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import time

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score, precision_score, \
    recall_score, plot_confusion_matrix, classification_report, plot_precision_recall_curve
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, LogisticRegressionCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, \
    HistGradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from data_reading_writing import read_data_and_labels


def export_model(model, model_name):
    if not os.path.exists('Models_Trained'):
        os.mkdir('Models_Trained')
    filename = 'Models_Trained/' + model_name + '.sav'
    pickle.dump(model, open(filename, 'wb'))
    print("Model saved in", filename)
    return None


def export_model_stats_json(model_dict, model_name, data_dict):
    if not os.path.exists('Model_Statistics'):
        os.mkdir('Model_Statistics')
    rel_file_path = 'Model_Statistics/' + model_name + '.json'
    del model_dict['model']
    model_dict['model_stats_train']['conf_matrix'] = [int(k) for k in
                                                      model_dict['model_stats_train']['conf_matrix'].flatten()]
    model_dict['model_stats_test']['conf_matrix'] = [int(k) for k in
                                                     model_dict['model_stats_test']['conf_matrix'].flatten()]
    dict_params = {}
    dict_params.update(model_dict)
    dict_params.update(data_dict)
    with open(rel_file_path, 'w') as outfile:
        json.dump(dict_params, outfile, indent=4)
    print("Model statistics saved in", rel_file_path)
    return None


def export_model_stats_csv(model_dict, model_name, data_dict):
    if not os.path.exists('Model_Statistics'):
        os.mkdir('Model_Statistics')
    filename = 'Model_Statistics/Model_Statistics.csv'
    if not os.path.exists(filename):
        title_string = 'Model name,Model_params,TRAIN Accuracy,Acc. Balanced,Precision,Recall,F1 Score,TEST Accuracy,Acc. Balanced,Precision,Recall,F1 Score,'
        for i in data_dict.keys():
            title_string = title_string + str(i) + ','
        title_string = title_string + '\n'
        with open(filename, 'w') as initfile:
            initfile.write(title_string)
        initfile.close()

    model_string = model_name + ',' + str(model_dict['model_params']).replace(',', '') + ','
    for key, model_value in model_dict['model_stats_train'].items():
        if key == 'conf_matrix':
            continue
        model_string = model_string + str(model_value) + ','
    for key, model_value in model_dict['model_stats_test'].items():
        if key == 'conf_matrix':
            continue
        model_string = model_string + str(model_value) + ','
    for data_value in data_dict.values():
        model_string = model_string + str(data_value).replace(',', '') + ','
    model_string = model_string + '\n'
    with open(filename, 'a') as outfile:
        outfile.write(model_string)
        outfile.close()

    print("Model statistics appended to", filename)
    return None


def read_model_stats_json(stats_path):
    with open(stats_path) as infile:
        stats_dict = json.load(infile)
    stats_dict['model_stats_train']['conf_matrix'] = np.reshape(stats_dict['model_stats_train']['conf_matrix'], (2, 2))
    stats_dict['model_stats_test']['conf_matrix'] = np.reshape(stats_dict['model_stats_test']['conf_matrix'], (2, 2))
    return stats_dict


def train_model(model, X_train, y_train):
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print(f'Training time: {(end_time - start_time)/60:.1f} minutes')
    return model


def evaluate_model(model, X, y):
    start_time = time.time()
    y_pred = model.predict(X)
    stats_dict = {'conf_matrix': confusion_matrix(y, y_pred), 'acc': accuracy_score(y, y_pred),
                  'acc_balanced': balanced_accuracy_score(y, y_pred), 'prec': precision_score(y, y_pred),
                  'rcll': recall_score(y, y_pred), 'f1_scr': f1_score(y, y_pred)}
    end_time = time.time()
    print(f'Evaluating time: {(end_time - start_time) / 60:.1f} minutes')
    return stats_dict


def get_name_index(model_name):
    idx = 0
    if os.path.exists('Model_Statistics'):
        model_paths = Path('Model_Statistics/').rglob(model_name + '*.json')
        list_model_paths = [str(path) for path in model_paths]
        idx = len(list_model_paths)
    return idx


def train_and_test_modelgroup(modelgroup, modelgroup_name, data, labels, preprocessing_params, test_size):
    index = get_name_index(modelgroup_name)
    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        labels,
                                                        test_size=test_size,
                                                        random_state=42,
                                                        stratify=labels)

    dict_data = {'training_size': y_train.size, 'training_nb_mites': int(np.sum(y_train)), 'test_size': y_test.size,
                 'test_nb_mites': int(np.sum(y_test)), 'feature_size': X_train.shape[1]}
    dict_data.update(preprocessing_params)

    for i in range(0, len(modelgroup)):
        model_name = modelgroup_name + '_' + str(index + i)
        dict_model = {'model': modelgroup[i], 'model_params': modelgroup[i].get_params()}

        dict_model['model'] = train_model(dict_model['model'], X_train, y_train)
        dict_model['model_stats_train'] = evaluate_model(dict_model['model'], X_train, y_train)
        dict_model['model_stats_test'] = evaluate_model(dict_model['model'], X_test, y_test)

        export_model(dict_model['model'], model_name)
        export_model_stats_json(dict_model, model_name, dict_data)
        export_model_stats_csv(dict_model, model_name, dict_data)
    return None


def train_and_test_model_selection(model_selection, folder_path, preprocessing_params, test_size):
    models = define_models(model_selection)
    data, labels = read_data_and_labels(folder_path, preprocessing_params)

    for key, value in models.items():
        train_and_test_modelgroup(value, key, data, labels, preprocessing_params, test_size)
    return None


def define_models(model_selection):
    log_reg_models = [LogisticRegression(penalty='none', max_iter=200, class_weight='balanced'),
                      LogisticRegression(penalty='l2', C=1.0, max_iter=200, class_weight='balanced'),
                      LogisticRegression(penalty='l1', C=1.0, max_iter=200, solver='saga', class_weight='balanced'),
                      LogisticRegression(penalty='elasticnet', C=1.0, solver='saga', l1_ratio=0.1,
                                         class_weight='balanced'),
                      LogisticRegression(penalty='l2', C=0.1, max_iter=200, class_weight='balanced'),
                      LogisticRegression(penalty='l1', C=0.1, max_iter=200, solver='saga', class_weight='balanced'),
                      LogisticRegression(penalty='elasticnet', C=0.1, solver='saga', l1_ratio=0.1,
                                         class_weight='balanced'),
                      LogisticRegression(penalty='l2', C=0.01, max_iter=200, class_weight='balanced'),
                      LogisticRegression(penalty='l1', C=0.01, max_iter=200, solver='saga', class_weight='balanced'),
                      LogisticRegression(penalty='elasticnet', C=0.01, solver='saga', l1_ratio=0.1,
                                         class_weight='balanced'),
                      LogisticRegression(penalty='l2', C=0.001, max_iter=200, class_weight='balanced'),
                      LogisticRegression(penalty='l1', C=0.001, max_iter=200, solver='saga', class_weight='balanced'),
                      LogisticRegression(penalty='elasticnet', C=0.001, solver='saga', l1_ratio=0.1,
                                         class_weight='balanced')]

    sgd_models = [SGDClassifier(penalty='l2', alpha=0.01, class_weight='balanced'),
                  SGDClassifier(penalty='l2', alpha=0.5, class_weight='balanced'),
                  SGDClassifier(penalty='l2', alpha=2.0, class_weight='balanced')]

    ridge_class_models = [RidgeClassifier(alpha=10.0, normalize=True, class_weight='balanced'),
                          RidgeClassifier(alpha=50.0, normalize=True, class_weight='balanced'),
                          RidgeClassifier(alpha=100.0, normalize=True, class_weight='balanced')]

    decision_tree_models = [DecisionTreeClassifier(max_depth=10, max_features='sqrt', class_weight='balanced'),
                            DecisionTreeClassifier(max_depth=100, max_features='sqrt', class_weight='balanced'),
                            DecisionTreeClassifier(max_features='sqrt', class_weight='balanced')]

    random_forest_models = [
        RandomForestClassifier(n_estimators=20, max_depth=3, max_features='sqrt', class_weight='balanced'),
        RandomForestClassifier(n_estimators=20, max_depth=10, max_features='sqrt', class_weight='balanced'),
        RandomForestClassifier(n_estimators=20, max_depth=100, max_features='sqrt', class_weight='balanced')]

    svm_models = [LinearSVC(penalty='l2', dual=False, C=1.0, class_weight='balanced'),
                  LinearSVC(penalty='l2', dual=False, C=0.1, class_weight='balanced'),
                  LinearSVC(penalty='l1', dual=False, C=1.0, class_weight='balanced'),
                  LinearSVC(penalty='l1', dual=False, C=0.1, class_weight='balanced'),
                  SVC(C=1.0, kernel='linear', class_weight='balanced'),
                  SVC(C=0.1, class_weight='balanced'), SVC(class_weight='balanced')]

    naive_bayes_models = [GaussianNB()]

    ada_boost_models = [AdaBoostClassifier(n_estimators=50), AdaBoostClassifier(n_estimators=100)]

    histogram_boost_models = [HistGradientBoostingClassifier(), HistGradientBoostingClassifier(l2_regularization=1.0),
                              HistGradientBoostingClassifier(l2_regularization=5.0)]

    gradient_boost_models = [GradientBoostingClassifier(), GradientBoostingClassifier(max_features='sqrt'),
                             GradientBoostingClassifier(max_features='log2')]

    log_reg_cv_models = [
        LogisticRegressionCV(Cs=[0.0001, 0.001, 0.01, 0.1, 1], max_iter=200, penalty='l2', class_weight='balanced')]

    models = {'log_reg': log_reg_models, 'sgd': sgd_models, 'ridge_class': ridge_class_models,
              'decision_tree': decision_tree_models, 'random_forest': random_forest_models, 'svm': svm_models,
              'naive_bayes': naive_bayes_models, 'ada_boost': ada_boost_models,
              'histogram_boost': histogram_boost_models, 'gradient_boost': gradient_boost_models,
              'log_reg_cv': log_reg_cv_models}

    for key, value in model_selection.items():
        if not value:
            models.pop(key)

    return models


def read_models(model_list):
    model_dict = {}
    for name in model_list:
        model = pickle.load(open('Models_Trained/' + name + '.sav', 'rb'))
        model_dict[name] = model
    return model_dict


def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Non-mite', 'Mite']))
    plot_confusion_matrix(model, X_test, y_test, display_labels=['Non-mite', 'Mite'])
    plot_precision_recall_curve(model, X_test, y_test)
    plt.show()
    return None


def get_feature_dims(model_dict):
    feature_dims = {}
    for key, value in model_dict.items():
        model_type = key[0:key.rfind('_')]
        if model_type in ['log_reg', 'sgd', 'ridge_class', 'log_reg_cv']:
            feature_dims[key] = value.coef_.shape[1]
        elif model_type in ['svm']:
            feature_dims[key] = value.support_vectors_.shape[1]
        elif model_type in ['naive_bayes']:
            feature_dims[key] = value.theta_.shape[1]
        elif model_type in ['ada_boost', 'gradien_boost']:
            feature_dims[key] = value.feature_importance_.shape[0]
        elif model_type in ['histogram_boost']:
            feature_dims[key] = value.is_categorical_.shape[0]
        else:
            feature_dims[key] = value.n_features_
    return feature_dims

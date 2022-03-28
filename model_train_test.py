import os
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from collections import OrderedDict

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score, precision_score, \
    recall_score, roc_auc_score, plot_confusion_matrix, classification_report, plot_precision_recall_curve, \
    RocCurveDisplay, PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from typing import Union
import lightgbm as lgb

from data_reading_writing import read_data_and_labels
from data_handling import split_and_sample_data, compute_prior_weight
from roc_precrcll_curves import compute_metric_curve


def export_model(model: object, model_name: str) -> None:
    if not os.path.exists('Models_Trained'):
        os.mkdir('Models_Trained')
    rel_file_path = 'Models_Trained/' + model_name + '.sav'
    pickle.dump(model, open(rel_file_path, 'wb'))
    print("Model saved in", rel_file_path)
    return None


def export_model_stats_json(model_dict: dict, model_name: str, data_dict: dict) -> None:
    if not os.path.exists('Training_Statistics'):
        os.mkdir('Training_Statistics')
    rel_file_path = 'Training_Statistics/' + model_name + '.json'
    del model_dict['model']
    model_dict['model_stats_train']['conf_matrix'] = [int(k) for k in
                                                      model_dict['model_stats_train']['conf_matrix'].flatten()]
    model_dict['model_stats_test']['conf_matrix'] = [int(k) for k in
                                                     model_dict['model_stats_test']['conf_matrix'].flatten()]
    dict_params = OrderedDict({})
    dict_params.update(model_dict)
    dict_params.update(data_dict)
    with open(rel_file_path, 'w') as outfile:
        json.dump(dict_params, outfile, indent=4)
    print("Model statistics saved in", rel_file_path)
    return None


def export_model_training_stats_csv(model_dict: dict, model_name: str, data_dict: dict) -> None:
    if not os.path.exists('Training_Statistics'):
        os.mkdir('Training_Statistics')
    filename = 'Training_Statistics/Model_Statistics.csv'
    if not os.path.exists(filename):
        title_string = 'Model name,Model_params,TRAIN Accuracy,Acc. Balanced,ROCAUC,Precision,Recall,F1 Score,TEST Accuracy,Acc. Balanced,ROCAUC,Precision,Recall,F1 Score,'
        for i in data_dict.keys():
            title_string = title_string + str(i) + ','
        title_string = title_string + '\n'
        with open(filename, 'w') as initfile:
            initfile.write(title_string)

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
    return None


def read_model_stats_json(stats_path: str) -> dict:
    with open(stats_path) as infile:
        stats_dict = json.load(infile)
    stats_dict['model_stats_train']['conf_matrix'] = np.reshape(stats_dict['model_stats_train']['conf_matrix'], (2, 2))
    stats_dict['model_stats_test']['conf_matrix'] = np.reshape(stats_dict['model_stats_test']['conf_matrix'], (2, 2))
    return stats_dict


def train_model(model: object, X_train: np.ndarray, y_train: np.ndarray) -> object:
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print('Training time: {:.0f}min {:.0f}s'.format((end_time - start_time) / 60, (end_time - start_time) % 60))
    return model


def evaluate_model(model: object, X: np.ndarray, y: np.ndarray, paths: list, prior_mite: float,
                   prior_no_mite: float) -> (dict, list, list):
    start_time = time.time()
    try:
        y_probs = model.predict_proba(X)
        y_probs[:, 1] = y_probs[:, 1] * prior_mite
        y_probs[:, 0] = y_probs[:, 0] * prior_no_mite
        sum_normalize = np.sum(y_probs, axis=1)
        y_probs = y_probs / sum_normalize[:, np.newaxis]
        y_pred = np.where(y_probs[:, 1] <= 0.5, 0, 1)
    except AttributeError:
        y_pred = np.around(model.predict(X))
        print('No probabilistic model for {} available; working with predictions instead.'.format(model))
    end_time = time.time()
    print('Evaluating time: {:.0f}min {:.0f}s'.format((end_time - start_time) / 60, (end_time - start_time) % 60))
    stats_dict = OrderedDict([('conf_matrix', confusion_matrix(y, y_pred)), ('acc', accuracy_score(y, y_pred)),
                              ('acc_balanced', balanced_accuracy_score(y, y_pred)),
                              ('prec', precision_score(y, y_pred)), ('rcll', recall_score(y, y_pred)),
                              ('f1_scr', f1_score(y, y_pred)), ('roc', roc_auc_score(y, y_pred))])
    if paths is not None:
        misclassified_imgs, true_pos_imgs = list_fp_fn_tp_images(y, y_pred, paths)
    else:
        misclassified_imgs = None
        true_pos_imgs = None
    return stats_dict, misclassified_imgs, true_pos_imgs


def export_model_evaluation_stats_json(stats_dict: dict, model_name: str) -> None:
    if not os.path.exists('Evaluation_Images'):
        os.mkdir('Evaluation_Images')
    if not os.path.exists('Evaluation_Images/' + model_name):
        os.mkdir('Evaluation_Images/' + model_name)
    rel_file_path = 'Evaluation_Images/' + model_name + '/Statistics.json'
    stats_dict.pop('acc')
    stats_dict.pop('acc_balanced')
    stats_dict.pop('f1_scr')
    conf_matrix = stats_dict['conf_matrix']
    stats_dict.pop('conf_matrix')
    stats_dict['Candidates'] = int(np.sum(conf_matrix))
    stats_dict['Mites'] = int(conf_matrix[1, 0] + conf_matrix[1, 1])
    stats_dict['true_pos'] = int(conf_matrix[1, 1])
    stats_dict['false_neg'] = int(conf_matrix[1, 0])
    stats_dict['false_pos'] = int(conf_matrix[0, 1])
    with open(rel_file_path, 'w') as outfile:
        json.dump(stats_dict, outfile, indent=4)
    print("Model statistics saved in", rel_file_path)
    return None


def evaluate_trained_model(path_test_data: str, data_params: dict, path_trained_model: str, model_name: str) -> None:
    try:
        model = pickle.load(open(path_trained_model, 'rb'))
    except pickle.UnpicklingError:
        model = lgb.Booster(model_file=path_trained_model)
    except RuntimeError:
        'Could no load any model for {}'.format(path_trained_model)
    X_test, y_test, paths_images = read_data_and_labels(path_test_data, data_params)
    # To do: prior weight cannot be computed since training data is not given
    stats_dict, misclassified_imgs, true_pos_imgs = evaluate_model(model, X_test, y_test, paths_images, prior_mite=1.0,
                                                                   prior_no_mite=1.0)
    export_evaluation_images_model(misclassified_imgs, true_pos_imgs, model_name, 'Evaluation')
    export_model_evaluation_stats_json(stats_dict, model_name)

    metrics = {'ROC': RocCurveDisplay, 'Precision-Recall': PrecisionRecallDisplay}
    for name, metric in metrics.items():
        fig = compute_metric_curve(metric, name, model, X_test, y_test)
        plt.savefig('Evaluation_Images/' + model_name + '/' + name + '.pdf')
        plt.show()

    try:
        plot_confusion_matrix(model, X_test, y_test)
        plt.show()
    except ValueError:
        y_pred = np.around(model.predict(X_test))
        print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    return None


def list_fp_fn_tp_images(y_true: np.ndarray, y_pred: np.ndarray, paths_images: list) -> (list, list):
    misclassified_imgs = paths_images[y_true + y_pred == 1]
    correct_imgs = paths_images[y_true + y_pred == 2]
    return misclassified_imgs, correct_imgs


def export_evaluation_images_model(misclassified_images: list, true_pos_images: list, model_name: str,
                                   train_test: str) -> None:
    model_dir = 'Evaluation_Images/' + model_name
    if not os.path.exists('Evaluation_Images'):
        os.mkdir('Evaluation_Images')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    dir_misclassified = model_dir + '/' + train_test + '_Misclassified/'
    export_images(misclassified_images, dir_misclassified)
    dir_true_pos = model_dir + '/' + train_test + '_TruePos/'
    export_images(true_pos_images, dir_true_pos)
    print('Exported evaluation images to {}'.format(model_dir))
    return None


def export_images(images_list: list, export_directory: str) -> None:
    if not os.path.exists(export_directory):
        os.mkdir(export_directory)
    filename_list = export_directory + 'image_list.txt'
    np.savetxt(filename_list, X=np.sort(images_list), delimiter=' ', fmt='%s')
    for path in images_list:
        path = path.replace('(', '\(')
        path = path.replace(')', '\)')
        os.system('cp {} ./{}'.format(path, export_directory))
    return None


def get_name_index(model_name: str, folder_name: str, file_format: str) -> int:
    idx = 0
    if os.path.exists(folder_name):
        list_model_paths = [str(path) for path in Path(folder_name).rglob(model_name + '*.' + file_format)]
        idx = len(list_model_paths)
    return idx


def train_and_test_modelgroup(modelgroup: list, modelgroup_name: str, X_train: np.ndarray, X_test: np.ndarray,
                              y_train: np.ndarray, y_test: np.ndarray, paths_train: Union[None, list],
                              paths_test: Union[None, list], data_params: dict, prior_mite: float,
                              prior_no_mite: float) -> None:
    index = get_name_index(modelgroup_name, 'Training_Statistics/', 'json')
    dict_data = OrderedDict([('training_size', y_train.size), ('training_nb_mites', int(np.sum(y_train))),
                             ('test_size', y_test.size), ('test_nb_mites', int(np.sum(y_test))),
                             ('feature_size', X_train.shape[1])])
    dict_data.update(data_params)
    for i in range(0, len(modelgroup)):
        model_name = modelgroup_name + '_' + str(index + i)
        dict_model = OrderedDict([('model', modelgroup[i]), ('model_params', modelgroup[i].get_params())])
        dict_model['model'] = train_model(dict_model['model'], X_train, y_train)
        dict_model['model_stats_train'], _, _ = evaluate_model(dict_model['model'], X_train, y_train, paths_train,
                                                               prior_mite, prior_no_mite)
        dict_model['model_stats_test'], _, _ = evaluate_model(dict_model['model'], X_test, y_test, paths_test,
                                                              prior_mite, prior_no_mite)
        # export_model(dict_model['model'], model_name)
        export_model_stats_json(dict_model, model_name, dict_data)
        export_model_training_stats_csv(dict_model, model_name, dict_data)

    return None


def train_and_test_model_selection(model_selection: dict, folder_path: str, data_params: dict, test_size: float,
                                   class_weight: Union[str, None], reweight_posterior: bool) -> None:
    models = define_models(model_selection, class_weight)

    data, labels, paths_images = read_data_and_labels(folder_path, data_params)
    X_train, X_test, y_train, y_test, _, _ = split_and_sample_data(data=data,
                                                                   labels=labels,
                                                                   paths_imgs=paths_images,
                                                                   test_size=test_size)
    del data
    if reweight_posterior:
        prior_mite, prior_no_mite = compute_prior_weight(np.array(labels), y_train)
    else:
        prior_mite, prior_no_mite = None, None
    for key, value in models.items():
        train_and_test_modelgroup(modelgroup=value,
                                  modelgroup_name=key,
                                  X_train=X_train,
                                  X_test=X_test,
                                  y_train=y_train,
                                  y_test=y_test,
                                  paths_train=None,
                                  paths_test=None,
                                  data_params=data_params,
                                  prior_mite=prior_mite,
                                  prior_no_mite=prior_no_mite)
    return None


def define_models(model_selection: dict, class_weight: Union[None, str]) -> dict:
    log_reg_models = [LogisticRegression(penalty='none', max_iter=200, class_weight=class_weight),
                      LogisticRegression(penalty='l2', C=10.0, max_iter=200, class_weight=class_weight),
                      LogisticRegression(penalty='l1', C=10.0, max_iter=200, solver='saga', class_weight=class_weight),
                      LogisticRegression(penalty='elasticnet', C=10.0, solver='saga', l1_ratio=0.1,
                                         class_weight=class_weight),
                      LogisticRegression(penalty='l2', C=1.0, max_iter=200, class_weight=class_weight),
                      LogisticRegression(penalty='l1', C=1.0, max_iter=200, solver='saga', class_weight=class_weight),
                      LogisticRegression(penalty='elasticnet', C=1.0, solver='saga', l1_ratio=0.1,
                                         class_weight=class_weight),
                      LogisticRegression(penalty='l2', C=0.1, max_iter=200, class_weight=class_weight),
                      LogisticRegression(penalty='l1', C=0.1, max_iter=200, solver='saga', class_weight=class_weight),
                      LogisticRegression(penalty='elasticnet', C=0.1, solver='saga', l1_ratio=0.1,
                                         class_weight=class_weight),
                      LogisticRegression(penalty='l2', C=0.01, max_iter=200, class_weight=class_weight),
                      LogisticRegression(penalty='l1', C=0.01, max_iter=200, solver='saga', class_weight=class_weight),
                      LogisticRegression(penalty='elasticnet', C=0.01, solver='saga', l1_ratio=0.1,
                                         class_weight=class_weight)]

    sgd_models = [SGDClassifier(penalty='l2', alpha=0.1, class_weight=class_weight),
                  SGDClassifier(penalty='l2', alpha=1.0, class_weight=class_weight),
                  SGDClassifier(penalty='l2', alpha=10.0, class_weight=class_weight)]

    ridge_class_models = [RidgeClassifier(alpha=0.01, normalize=True, max_iter=None, class_weight=class_weight),
                          RidgeClassifier(alpha=0.1, normalize=True, max_iter=None, class_weight=class_weight),
                          RidgeClassifier(alpha=1.0, normalize=True, max_iter=None, class_weight=class_weight),
                          RidgeClassifier(alpha=10.0, normalize=True, max_iter=None, class_weight=class_weight),
                          RidgeClassifier(alpha=100.0, normalize=True, max_iter=None, class_weight=class_weight)]

    decision_tree_models = [DecisionTreeClassifier(max_depth=10, max_features='sqrt', class_weight=class_weight),
                            DecisionTreeClassifier(max_depth=100, max_features='sqrt', class_weight=class_weight),
                            DecisionTreeClassifier(max_features='sqrt', class_weight=class_weight)]

    random_forest_models = [
        RandomForestClassifier(n_estimators=20, max_depth=3, max_features='sqrt', class_weight=class_weight),
        RandomForestClassifier(n_estimators=20, max_depth=10, max_features='sqrt', class_weight=class_weight),
        RandomForestClassifier(n_estimators=20, max_depth=100, max_features='sqrt', class_weight=class_weight),
        RandomForestClassifier(n_estimators=50, max_depth=3, max_features='sqrt', class_weight=class_weight),
        RandomForestClassifier(n_estimators=50, max_depth=10, max_features='sqrt', class_weight=class_weight),
        RandomForestClassifier(n_estimators=50, max_depth=100, max_features='sqrt', class_weight=class_weight),
        RandomForestClassifier(n_estimators=100, max_depth=3, max_features='sqrt', class_weight=class_weight),
        RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt', class_weight=class_weight),
        RandomForestClassifier(n_estimators=100, max_depth=100, max_features='sqrt', class_weight=class_weight)]

    l_svm_models = [LinearSVC(penalty='l2', dual=False, C=1.0, max_iter=500, class_weight=class_weight),
                    LinearSVC(penalty='l2', dual=False, C=0.1, max_iter=500, class_weight=class_weight),
                    LinearSVC(penalty='l1', dual=False, C=1.0, max_iter=500, class_weight=class_weight),
                    LinearSVC(penalty='l1', dual=False, C=0.1, max_iter=500, class_weight=class_weight)]

    nl_svm_models = [SVC(C=0.1, class_weight=class_weight),
                     SVC(C=1.0, class_weight=class_weight),
                     SVC(C=10.0, class_weight=class_weight),
                     SVC(C=0.1, kernel='poly', class_weight=class_weight),
                     SVC(C=0.1, kernel='poly', class_weight=class_weight),
                     SVC(C=10.0, kernel='poly', class_weight=class_weight)]

    naive_bayes_models = [GaussianNB()]

    ada_boost_models = [AdaBoostClassifier(n_estimators=10),
                        AdaBoostClassifier(n_estimators=50),
                        AdaBoostClassifier(n_estimators=100),
                        AdaBoostClassifier(n_estimators=200),
                        AdaBoostClassifier(n_estimators=10, learning_rate=0.1),
                        AdaBoostClassifier(n_estimators=50, learning_rate=0.1),
                        AdaBoostClassifier(n_estimators=100, learning_rate=0.1),
                        AdaBoostClassifier(n_estimators=200, learning_rate=0.1)]

    histogram_boost_models = [LGBMClassifier(n_estimators=10, class_weight=class_weight),
                              LGBMClassifier(n_estimators=10, class_weight=class_weight, reg_lambda=0.1),
                              LGBMClassifier(n_estimators=10, class_weight=class_weight, reg_lambda=1.0),
                              LGBMClassifier(n_estimators=10, class_weight=class_weight, reg_lambda=10.0),
                              LGBMClassifier(n_estimators=100, class_weight=class_weight),
                              LGBMClassifier(n_estimators=100, class_weight=class_weight, reg_lambda=0.1),
                              LGBMClassifier(n_estimators=100, class_weight=class_weight, reg_lambda=1.0),
                              LGBMClassifier(n_estimators=100, class_weight=class_weight, reg_lambda=10.0),
                              LGBMClassifier(n_estimators=200, class_weight=class_weight),
                              LGBMClassifier(n_estimators=200, class_weight=class_weight, reg_lambda=0.1),
                              LGBMClassifier(n_estimators=200, class_weight=class_weight, reg_lambda=1.0),
                              LGBMClassifier(n_estimators=200, class_weight=class_weight, reg_lambda=10.0)
                              ]

    gradient_boost_models = [GradientBoostingClassifier(n_estimators=10),
                             GradientBoostingClassifier(n_estimators=10, max_features='sqrt'),
                             GradientBoostingClassifier(n_estimators=10, max_features='log2'),
                             GradientBoostingClassifier(n_estimators=100),
                             GradientBoostingClassifier(n_estimators=100, max_features='sqrt'),
                             GradientBoostingClassifier(n_estimators=100, max_features='log2'),
                             GradientBoostingClassifier(n_estimators=200),
                             GradientBoostingClassifier(n_estimators=200, max_features='sqrt'),
                             GradientBoostingClassifier(n_estimators=200, max_features='log2')]

    handicraft_models = [LGBMClassifier(n_estimators=100, class_weight='balanced', num_leaves=5),
                         LGBMClassifier(n_estimators=100, class_weight='balanced', reg_lambda=0.1, num_leaves=5),
                         LGBMClassifier(n_estimators=100, class_weight='balanced', reg_lambda=1.0, num_leaves=5),
                         LGBMClassifier(n_estimators=100, class_weight='balanced', reg_lambda=10.0, num_leaves=5)]

    models = OrderedDict([('log_reg', log_reg_models), ('sgd', sgd_models), ('ridge_class', ridge_class_models),
                          ('decision_tree', decision_tree_models), ('random_forest', random_forest_models),
                          ('l_svm', l_svm_models), ('nl_svm', nl_svm_models), ('naive_bayes', naive_bayes_models),
                          ('ada_boost', ada_boost_models), ('histogram_boost', histogram_boost_models),
                          ('gradient_boost', gradient_boost_models), ('handicraft', handicraft_models)])

    for key, value in model_selection.items():
        if not value:
            models.pop(key)

    return models


def read_models(model_list: list) -> dict:
    model_dict = OrderedDict({})
    for name in model_list:
        model = pickle.load(open('Models_Trained/' + name + '.sav', 'rb'))
        model_dict[name] = model
    return model_dict


def test_model(model: object, X_test: np.ndarray, y_test: np.ndarray) -> None:
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Non-mite', 'Mite']))
    plot_confusion_matrix(model, X_test, y_test, display_labels=['Non-mite', 'Mite'])
    plot_precision_recall_curve(model, X_test, y_test)
    plt.show()
    return None


def get_feature_dims(model_dict: dict) -> dict:
    feature_dims = OrderedDict({})
    for key, value in model_dict.items():
        model_type = key[0:key.rfind('_')]
        if model_type in ['log_reg', 'sgd', 'ridge_class', 'log_reg_cv']:
            feature_dims[key] = value.coef_.shape[1]
        elif model_type in ['svm']:
            feature_dims[key] = value.support_vectors_.shape[1]
        elif model_type in ['naive_bayes']:
            feature_dims[key] = value.theta_.shape[1]
        elif model_type in ['ada_boost', 'gradient_boost']:
            feature_dims[key] = value.feature_importance_.shape[0]
        elif model_type in ['histogram_boost']:
            feature_dims[key] = value.is_categorical_.shape[0]
        else:
            feature_dims[key] = value.n_features_
    return feature_dims

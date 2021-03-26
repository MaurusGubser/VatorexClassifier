import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB

from data_reading_writing import read_data_and_labels
from data_handling import downsize_false_candidates
from model_train_test import get_name_index


def train_sequential_model(model_0, model_1, data, labels):
    start_time = time.time()
    model_0.fit(data, labels)
    y_1 = model_0.predict(data)
    data_2 = data[y_1 == 1]
    labels_2 = labels[y_1 == 1]
    model_1.fit(data_2, labels_2)
    end_time = time.time()
    print('Training time: {:.0f}min {:.0f}s'.format((end_time - start_time) / 60, (end_time - start_time) % 60))
    return model_0, model_1


def predict_sequential_model(model_0, model_1, data):
    y_0 = model_0.predict(data)
    data_1 = data[y_0 == 1]
    y_1 = model_1.predict(data_1)
    y_pred = np.zeros(y_0.shape, dtype=np.int8)
    j = 0
    for i in range(0, y_0.size):
        if y_0[i] == 1:
            y_pred[i] = y_1[j]
            j += 1
    return y_pred, y_0, y_1


def compute_mites_model_1(y_true, y_0):
    y_ones = (y_true + y_0 == 2)
    return int(np.sum(y_ones))


def evaluate_sequential_model(model_0, model_1, X, y):
    start_time = time.time()
    y_pred, y_0, y_1 = predict_sequential_model(model_0, model_1, X)
    """
    stats_dict_1 = OrderedDict([('conf_matrix', confusion_matrix(y, y_0)), ('acc', accuracy_score(y, y_0)),
                                ('acc_balanced', balanced_accuracy_score(y, y_0)),
                                ('prec', precision_score(y, y_0)), ('rcll', recall_score(y, y_0)),
                                ('f1_scr', f1_score(y, y_0))])
    """
    stats_dict_seq = OrderedDict([('conf_matrix', confusion_matrix(y, y_pred)), ('acc', accuracy_score(y, y_pred)),
                                  ('acc_balanced', balanced_accuracy_score(y, y_pred)),
                                  ('prec', precision_score(y, y_pred)), ('rcll', recall_score(y, y_pred)),
                                  ('f1_scr', f1_score(y, y_pred)), ('Model_1 cand', y_1.size),
                                  ('Model_1 true mites', compute_mites_model_1(y, y_0)),
                                  ('Model_1 pred mites', int(np.sum(y_1)))])

    end_time = time.time()
    print('Evaluating time: {:.0f}min {:.0f}s'.format((end_time - start_time) / 60, (end_time - start_time) % 60))
    return stats_dict_seq


def export_sequential_model_stats_json(model_dict, model_name, data_dict):
    if not os.path.exists('Sequential_Model_Statistics'):
        os.mkdir('Sequential_Model_Statistics')
    rel_file_path = 'Sequential_Model_Statistics/' + model_name + '.json'
    del model_dict['model_0']
    del model_dict['model_1']
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


def export_sequential_model_stats_csv(model_dict, model_name, data_dict):
    if not os.path.exists('Sequential_Model_Statistics'):
        os.mkdir('Sequential_Model_Statistics')
    filename = 'Sequential_Model_Statistics/Sequential_Model_Statistics.csv'
    if not os.path.exists(filename):
        title_string = 'Model name,Model_0_params,Model_1_params,TRAIN Accuracy,Acc. Balanced,Precision,Recall,F1 Score,Model_1 cand,Model_1 true mites,Model_1 pred mites,TEST Accuracy,Acc. Balanced,Precision,Recall,F1 Score,Model_1 cand,Model_1 true mites,Model_1 pred mites,'
        for i in data_dict.keys():
            title_string = title_string + str(i) + ','
        title_string = title_string + '\n'
        with open(filename, 'w') as initfile:
            initfile.write(title_string)

    model_string = model_name + ',' + str(model_dict['model_0_params']).replace(',', '') + ',' + str(
        model_dict['model_1_params']).replace(',', '') + ','
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

    print("Model statistics of {} appended to {}".format(model_name, filename))
    return None


def get_name_sequential_index(model_name):
    idx = 0
    if os.path.exists('Sequential_Model_Statistics'):
        list_model_paths = [str(path) for path in Path('Sequential_Model_Statistics/').rglob(model_name + '*.json')]
        idx = len(list_model_paths)
    return idx


def train_and_test_sequential_models(sequential_models, folder_path, data_params, test_size):
    data, labels = read_data_and_labels(folder_path, data_params)
    data, labels = downsize_false_candidates(data, labels, data_params['percentage_true'])
    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        labels,
                                                        test_size=test_size,
                                                        random_state=42,
                                                        stratify=labels)
    del data

    for key, value in sequential_models.items():
        index = get_name_sequential_index(key)
        dict_data = OrderedDict([('training_size', y_train.size), ('training_nb_mites', int(np.sum(y_train))),
                                 ('test_size', y_test.size), ('test_nb_mites', int(np.sum(y_test))),
                                 ('feature_size', X_train.shape[1])])
        dict_data.update(data_params)

        model_name = key + '_' + str(index)
        dict_model = OrderedDict([('model_0', value[0]), ('model_1', value[1]),
                                  ('model_0_params', value[0].get_params()), ('model_1_params', value[1].get_params())])

        dict_model['model_0'], dict_model['model_1'] = train_sequential_model(dict_model['model_0'],
                                                                              dict_model['model_1'], X_train, y_train)
        dict_model['model_stats_train'] = evaluate_sequential_model(dict_model['model_0'], dict_model['model_1'],
                                                                    X_train, y_train)
        dict_model['model_stats_test'] = evaluate_sequential_model(dict_model['model_0'], dict_model['model_1'], X_test,
                                                                   y_test)

        # export_model(dict_model['model'], model_name)
        export_sequential_model_stats_json(dict_model, model_name, dict_data)
        export_sequential_model_stats_csv(dict_model, model_name, dict_data)

    return None


def define_sequential_models(model_names, models_rcll, models_prec):
    sequential_models = OrderedDict({})
    if len(model_names) != len(models_rcll) or len(model_names) != len(models_prec):
        raise AssertionError(
            'Lists names, models_rcll and models_prec have to be of the same length; have length {}, {}, {}'.format(
                len(model_names), len(models_rcll), len(models_prec)))
    for i in range(0, len(model_names)):
        sequential_models[model_names[i]] = [models_rcll[i], models_prec[i]]
    return sequential_models

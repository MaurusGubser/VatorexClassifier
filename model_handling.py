import os
from pathlib import Path
import pickle
import numpy as np
import json
import time
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score, \
    precision_score, recall_score


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
    model_dict['model_stats_train']['conf_matrix'] = [int(k) for k in model_dict['model_stats_train']['conf_matrix'].flatten()]
    model_dict['model_stats_test']['conf_matrix'] = [int(k) for k in model_dict['model_stats_test']['conf_matrix'].flatten()]
    dict = {}
    dict.update(model_dict)
    dict.update(data_dict)
    with open(rel_file_path, 'w') as outfile:
        json.dump(dict, outfile, indent=4)
    print("Model statistics saved in", rel_file_path)
    return None


def export_model_stats_csv(model_dict, model_name, data_dict):
    if not os.path.exists('Model_Statistics'):
        os.mkdir('Model_Statistics')
    filename = 'Model_Statistics/Model_Statistics.csv'
    if not os.path.exists(filename):
        title_string = 'Model name,Model_params,Train Accuracy,Acc. Balanced,Precision,Recall,F1 Score,Test Accuracy,Acc. Balanced,Precision,Recall,F1 Score,'
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


def read_model(model_path):
    model = pickle.load(open(model_path, 'rb'))
    return model


def read_model_stats_json(stats_path):
    with open(stats_path) as infile:
        stats_dict = json.load(infile)
    stats_dict['model_stats_train']['conf_matrix'] = np.reshape(stats_dict['model_stats_train']['conf_matrix'], (2, 2))
    stats_dict['model_stats_test']['conf_matrix'] = np.reshape(stats_dict['model_stats_test']['conf_matrix'], (2, 2))
    return stats_dict


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    stats_dict = {'conf_matrix': confusion_matrix(y, y_pred), 'acc': accuracy_score(y, y_pred),
                  'acc_balanced': balanced_accuracy_score(y, y_pred), 'prec': precision_score(y, y_pred),
                  'rcll': recall_score(y, y_pred), 'f1_scr': f1_score(y, y_pred)}
    return stats_dict


def get_name_index(model_name):
    idx = 0
    if os.path.exists('Model_Statistics'):
        model_paths = Path('Model_Statistics/').rglob(model_name + '*.json')
        list_model_paths = [str(path) for path in model_paths]
        idx = len(list_model_paths)
    return idx


def train_and_evaluate_modelgroup(modelgroup, modelgroup_name, data_params, preproc_params):
    index = get_name_index(modelgroup_name)
    training_folder_names = data_params['training_folder_names']
    X_train = data_params['X_train']
    y_train = data_params['y_train']
    test_folder_names = data_params['test_folder_names']
    X_test = data_params['X_test']
    y_test = data_params['y_test']

    dict_data = {'training_data': training_folder_names, 'training_size': y_train.size,
                 'training_nb_mites': int(np.sum(y_train)), 'test_data': test_folder_names, 'test_size': y_test.size,
                 'test_nb_mites': int(np.sum(y_test))}
    dict_data.update(preproc_params)

    for i in range(0, len(modelgroup)):
        model_name = modelgroup_name + '_' + str(index + i)
        dict_model = {'model': modelgroup[i], 'model_params': modelgroup[i].get_params()}

        start_time = time.time()
        dict_model['model'] = train_model(dict_model['model'], X_train, y_train)
        end_time = time.time()
        print('Training time {}: {:.1f} minutes'.format(model_name, (end_time - start_time) / 60))
        dict_model['model_stats_train'] = evaluate_model(dict_model['model'], X_train, y_train)
        dict_model['model_stats_test'] = evaluate_model(dict_model['model'], X_test, y_test)
        export_model(dict_model['model'], model_name)
        export_model_stats_json(dict_model, model_name, dict_data)
        export_model_stats_csv(dict_model, model_name, dict_data)
    return None

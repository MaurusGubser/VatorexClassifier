import os
from pathlib import Path
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score, \
    precision_score, recall_score


def export_model(model, model_name):
    if not os.path.exists('Models_Trained'):
        os.mkdir('Models_Trained')
    filename = 'Models_Trained/' + model_name + '.sav'
    pickle.dump(model, open(filename, 'wb'))
    print("Model saved in", filename)
    return None


def export_model_stats_json(model_dict, model_name):
    if not os.path.exists('Model_Statistics'):
        os.mkdir('Model_Statistics')
    filename = 'Model_Statistics/' + model_name + '.json'
    del model_dict['model']
    model_dict['model_stats']['conf_matrix'] = [int(k) for k in model_dict['model_stats']['conf_matrix'].flatten()]
    with open(filename, 'w') as outfile:
        json.dump(model_dict, outfile, indent=4)
    print("Model statistics saved in", filename)
    return None


def export_model_stats_csv(model_dict, model_name):
    if not os.path.exists('Model_Statistics'):
        os.mkdir('Model_Statistics')
    filename = 'Model_Statistics/Model_Statistics.csv'
    if not os.path.exists(filename):
        with open(filename, 'w') as initfile:
            initfile.write(
                'Model name, Model params, Training sets, Training size, Training nb mites, Test sets, Test size, Test nb mites, Normalize mean, Normalize std, Normalize histogram, acc, acc_balanced, prec, rcll, f1_scr \n')
    with open(filename, 'a') as outfile:
        outfile.write(model_name + ',')
        outfile.write(str(model_dict['model_params']).replace(',', '') + ',')
        outfile.write(str(model_dict['training_data']).replace(',', '') + ',')
        outfile.write(str(model_dict['training_size']) + ',')
        outfile.write(str(model_dict['training_nb_mites']) + ',')
        outfile.write(str(model_dict['test_data']).replace(',', '') + ',')
        outfile.write(str(model_dict['test_size']) + ',')
        outfile.write(str(model_dict['test_nb_mites']) + ',')
        outfile.write(str(model_dict['normalize_mean']) + ',')
        outfile.write(str(model_dict['normalize_std']) + ',')
        outfile.write(str(model_dict['normalize_hist']) + ',')
        outfile.write(str(model_dict['model_stats']['acc']) + ',')
        outfile.write(str(model_dict['model_stats']['acc_balanced']) + ',')
        outfile.write(str(model_dict['model_stats']['prec']) + ',')
        outfile.write(str(model_dict['model_stats']['rcll']) + ',')
        outfile.write(str(model_dict['model_stats']['f1_scr']) + '\n')
    print("Model statistics appended to", filename)
    return None


def read_model(model_path):
    model = pickle.load(open(model_path, 'rb'))
    return model


def read_model_stats_json(stats_path):
    with open(stats_path) as infile:
        stats_dict = json.load(infile)
    stats_dict['model_stats']['conf_matrix'] = np.reshape(stats_dict['model_stats']['conf_matrix'], (2, 2))
    return stats_dict


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    plot_confusion_matrix(model, X_test, y_test, display_labels=['Non-Mite', 'Mite'])
    plt.title('Confusion matrix of test set')
    stats_dict = {'conf_matrix': confusion_matrix(y_test, y_pred), 'acc': accuracy_score(y_test, y_pred),
                  'acc_balanced': balanced_accuracy_score(y_test, y_pred), 'prec': precision_score(y_test, y_pred),
                  'rcll': recall_score(y_test, y_pred), 'f1_scr': f1_score(y_test, y_pred)}
    return stats_dict


def get_name_index(model_name):
    idx = 0
    if os.path.exists('Model_Statistics'):
        model_paths = Path('Model_Statistics/').rglob(model_name + '.json')
        list_model_paths = [str(path) for path in model_paths]
        idx = len(list_model_paths)
    return idx

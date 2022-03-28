import json
import os

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
from glob import glob

from data_handling import split_and_sample_data
from data_reading_writing import get_folder_name, read_data_and_labels


def get_data_path(path: str) -> str:
    start = 0
    end = path.rfind("/")
    root_path = path[start:end + 1]
    return root_path


def train_lgbm(parameters: dict, data: np.ndarray, labels: np.ndarray, cv: int, export_path: str) -> lgb.Booster:
    train_data_lgb = lgb.Dataset(data, label=labels)
    cv_dict = lgb.cv(params=parameters, train_set=train_data_lgb, nfold=cv, return_cvbooster=True)
    model_lgb = cv_dict['cvbooster']
    model_lgb.save_model(export_path, num_iteration=model_lgb.best_iteration)
    model_lgb = model_lgb.boosters[model_lgb.best_iteration]
    return model_lgb


def train_sklearn(param_lgbm: dict, X_train: np.ndarray, y_train: np.ndarray) -> LGBMClassifier:
    model_sklearn = LGBMClassifier()
    for key, value in param_lgbm.items():
        model_sklearn.set_params(**{key: value})
    model_sklearn.fit(X_train, y_train)
    return model_sklearn


def model_params_to_str(params_model: dict) -> str:
    name = ''
    for key, value in params_model.items():
        name = name + '_' + str(key) + str(value)
    return name


def export_GUI_model(folder_path: str, data_params: dict, test_size: float, cv: int, param_lgbm: dict) -> None:
    data_params_str = get_folder_name(folder_path)
    model_params_str = model_params_to_str(param_lgbm)
    model_name = 'LightGBM_Model_Vatorex.txt'
    data, labels, paths_imgs = read_data_and_labels(folder_path, data_params)
    X_train, X_test, y_train, y_test, _, _ = split_and_sample_data(data=data,
                                                                   labels=labels,
                                                                   paths_imgs=paths_imgs,
                                                                   test_size=test_size)
    if not os.path.exists('GUI_Model_Export'):
        os.mkdir('GUI_Model_Export')
    subfolder_name = 'GUI_Model_Export/' + data_params_str + model_params_str
    if not os.path.exists(subfolder_name):
        os.mkdir(subfolder_name)
    export_path = subfolder_name + '/' + model_name
    with open(subfolder_name + '/Data_Parameters.json', 'w') as outfile:
        json.dump(data_params, outfile, indent=4)
    with open(subfolder_name + '/Model_Parameters.json', 'w') as outfile:
        json.dump(param_lgbm, outfile, indent=4)
    model_lgbm = train_lgbm(param_lgbm, X_train, y_train, cv, export_path)

    if test_size is not None:
        model_sklearn = train_sklearn(param_lgbm, X_train, y_train)
        y_pred_lgbm = np.around(model_lgbm.predict(X_test))
        y_pred_sklearn = model_sklearn.predict(X_test)
        conf_matrix_lgbm = confusion_matrix(y_test, y_pred_lgbm)
        print('LGBM confusion matrix\n', conf_matrix_lgbm)
        conf_matrix_sklearn = confusion_matrix(y_test, y_pred_sklearn)
        print('Sklearn confusion matrix\n', conf_matrix_sklearn)
    return None

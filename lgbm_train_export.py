import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, precision_recall_curve, plot_confusion_matrix

from data_handling import downsize_false_candidates
from data_reading_writing import load_data_and_labels


def get_data_path(path):
    start = 0
    end = path.rfind("/")
    root_path = path[start:end+1]
    return root_path


def train_lgbm(parameters, data, labels, cv, export_path):
    train_data_lgb = lgb.Dataset(data, label=labels)
    cv_dict = lgb.cv(params=parameters, train_set=train_data_lgb, nfold=cv, return_cvbooster=True)
    model_lgb = cv_dict['cvbooster']
    model_lgb.save_model(export_path, num_iteration=model_lgb.best_iteration)
    return None


def export_GUI_model(path_data, percentage_true, cv, param_lgbm, export_name):
    _, data, labels, path_images = load_data_and_labels(path_data)
    data, labels, _ = downsize_false_candidates(data, labels, path_images, percentage_true)
    export_path = get_data_path(path_data) + export_name
    train_lgbm(param_lgbm, data, labels, cv, export_path)
    return None

import lightgbm as lgb

from data_handling import split_and_sample_data
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


def export_GUI_model(path_data, undersampling_rate, oversampling_rate, cv, param_lgbm, export_name):
    _, data, labels, path_images = load_data_and_labels(path_data)
    data, _, labels, _, _, _ = split_and_sample_data(data=data,
                                                     labels=labels,
                                                     paths_imgs=path_images,
                                                     test_size=None,
                                                     undersampling_rate=undersampling_rate,
                                                     oversampling_rate=oversampling_rate)
    export_path = get_data_path(path_data) + export_name
    train_lgbm(param_lgbm, data, labels, cv, export_path)
    return None

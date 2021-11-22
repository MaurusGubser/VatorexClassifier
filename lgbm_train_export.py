import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
from typing import Union

from data_handling import split_and_sample_data
from data_reading_writing import load_data_and_labels


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


def export_GUI_model(path_data: str, undersampling_rate: Union[None, float], oversampling_rate: Union[None, float],
                     test_size: float, cv: int, param_lgbm: dict, export_name: str) -> None:
    _, data, labels, path_images = load_data_and_labels(path_data)
    X_train, X_test, y_train, y_test, _, _ = split_and_sample_data(data=data,
                                                                   labels=labels,
                                                                   paths_imgs=path_images,
                                                                   test_size=test_size,
                                                                   undersampling_rate=undersampling_rate,
                                                                   oversampling_rate=oversampling_rate)
    export_path = get_data_path(path_data) + export_name
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

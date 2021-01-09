from model_handling import *
from data_handling import *
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
import time

# Define some different models for classification
log_reg_models = [LogisticRegression(penalty='none', max_iter=200), LogisticRegression(penalty='l2', max_iter=200),
                  LogisticRegression(penalty='l2', max_iter=200, C=0.2),
                  LogisticRegression(penalty='l2', max_iter=200, C=2),
                  LogisticRegression(penalty='l2', max_iter=200, C=10),
                  LogisticRegression(penalty='l1', max_iter=200, solver='saga'),
                  LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.1),
                  LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.9)]

ridge_class_models = [RidgeClassifier(alpha=5.0), RidgeClassifier(alpha=1.0), RidgeClassifier(alpha=0.5),
                      RidgeClassifier(alpha=0.1), RidgeClassifier(alpha=0.01)]

random_forest_models = [RandomForestClassifier(n_estimators=200, class_weight='balanced'),
                        RandomForestClassifier(n_estimators=50), RandomForestClassifier(n_estimators=100),
                        RandomForestClassifier(n_estimators=200)]

ada_boost_models = [AdaBoostClassifier(n_estimators=50), AdaBoostClassifier(n_estimators=100),
                    AdaBoostClassifier(n_estimators=200)]

log_reg_cv_models = [LogisticRegressionCV(penalty='l2', max_iter=200),
                     LogisticRegressionCV(penalty='l1', max_iter=200, solver='saga'),
                     LogisticRegressionCV(penalty='elasticnet', max_iter=200, solver='saga',
                                          l1_ratios=[0.01, 0.1, 0.5, 0.9, 0.99])]

gradient_boosting_models = [GradientBoostingClassifier(n_estimators=50),
                            GradientBoostingClassifier(n_estimators=100),
                            GradientBoostingClassifier(n_estimators=200)]

if __name__ == '__main__':
    rel_path = "Training_Images/"
    images_paths = get_paths_of_image_folders(rel_path)

    # Choose training set and test set
    training_paths = images_paths[3:10]
    test_paths = images_paths[0:3]
    training_folder_names = [get_folder_name(path) for path in training_paths]
    test_folder_names = [get_folder_name(path) for path in test_paths]
    print("Train folders:", training_paths)
    print("Test folders:", test_paths)

    # Define data preprocessing options
    gray_scale = False
    normalize_hist = True
    with_image = True
    with_binary_patterns = False
    with_haar_features = False
    with_mean = True
    with_std = False

    preprocessing_parameters = {'gray_scale': gray_scale, 'normalize_hist': normalize_hist, 'with_image': with_image,
                                'with_binary_patterns': with_binary_patterns, 'with_haar_features': with_haar_features,
                                'with_mean': with_mean, 'with_std': with_std}
    X_train, y_train = prepare_data_and_labels(training_paths, preprocessing_parameters)
    X_test, y_test = prepare_data_and_labels(test_paths, preprocessing_parameters)
    data_parameters = {'training_folder_names': training_folder_names, 'X_train': X_train, 'y_train': y_train,
                       'test_folder_names': test_folder_names, 'X_test': X_test, 'y_test': y_test}

    # Define which models to train and evaluate; name describes the model type (log_reg, ridge_class, random_forest, ada_boost, etc)
    models = log_reg_models
    name = 'log_reg'

    train_and_evaluate_modelgroup(modelgroup=models, modelgroup_name=name, data_params=data_parameters,
                                  preproc_params=preprocessing_parameters)

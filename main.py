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
    path = "Training_Images/"
    images_paths = get_paths_image_folders(path)
    large_training_paths = images_paths[0:8] + images_paths[9:12]
    small_training_paths = images_paths[3:10]

    # Choose small or large training set and test set
    training_paths = small_training_paths
    test_paths = images_paths[2:3]
    training_folder_names = [get_folder_name(path) for path in training_paths]
    test_folder_names = [get_folder_name(path) for path in test_paths]
    print("Train folders:", training_paths)
    print("Test folders:", test_paths)

    # Choose preprocessing options
    normalize_mean = True
    normalize_std = False
    normalize_hist = True
    X_train, y_train = read_data(training_paths, normalize_mean=normalize_mean, normalize_std=normalize_std,
                                 normalize_hist=normalize_hist)
    X_test, y_test = read_data(test_paths, normalize_mean=normalize_mean, normalize_std=normalize_std,
                               normalize_hist=normalize_hist)

    # Define which models to train and evaluate; name describes the model type (log_reg, ridge_class, random_forest, ada_boost, etc)
    models = log_reg_models
    name = 'log_reg'
    index = get_name_index(name)

    for i in range(0, len(models)):
        model_name = name + '_' + str(index + i)
        dict_model = {'model': models[i], 'model_params': models[i].get_params(),
                      'training_data': training_folder_names, 'training_size': y_train.size,
                      'training_nb_mites': int(np.sum(y_train)),
                      'test_data': test_folder_names, 'test_size': y_test.size, 'test_nb_mites': int(np.sum(y_test)),
                      'normalize_mean': normalize_mean, 'normalize_std': normalize_std,
                      'normalize_hist': normalize_hist}
        start_time = time.time()
        dict_model['model'] = train_model(dict_model['model'], X_train, y_train)
        end_time = time.time()
        print('Training time {}: {:.1f} minutes'.format(model_name, (end_time - start_time) / 60))
        dict_model['model_stats'] = evaluate_model(dict_model['model'], X_test, y_test)
        export_model(dict_model['model'], model_name)
        export_model_stats_json(dict_model, model_name)
        export_model_stats_csv(dict_model, model_name)

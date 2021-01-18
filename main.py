from model_handling import *
from data_handling import *
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# Define some different models for classification
log_reg_models = [LogisticRegression(penalty='none', C=1.0, max_iter=200, class_weight='balanced'),
                  LogisticRegression(penalty='l2', C=1.0, max_iter=200, class_weight='balanced'),
                  LogisticRegression(penalty='l1', C=1.0, max_iter=200, solver='saga', class_weight='balanced'),
                  LogisticRegression(penalty='elasticnet', C=1.0, solver='saga', l1_ratio=0.1, class_weight='balanced')]

sgd_models = [SGDClassifier(penalty='l2', alpha=0.01, class_weight='balanced'),
              SGDClassifier(penalty='l2', alpha=0.8, class_weight='balanced'),
              SGDClassifier(penalty='l2', alpha=2.0, class_weight='balanced')]

ridge_class_models = [RidgeClassifier(alpha=5.0, normalize=True, class_weight='balanced'),
                      RidgeClassifier(alpha=1.0, normalize=True, class_weight='balanced'),
                      RidgeClassifier(alpha=0.01, normalize=True, class_weight='balanced')]

decision_tree_models = [DecisionTreeClassifier(max_depth=3, class_weight='balanced'),
                        DecisionTreeClassifier(max_depth=10, class_weight='balanced'),
                        DecisionTreeClassifier(max_depth=100, class_weight='balanced'),
                        DecisionTreeClassifier(class_weight='balanced')]

random_forest_models = [RandomForestClassifier(n_estimators=10, max_depth=3, max_features='sqrt', class_weight='balanced'),
                        RandomForestClassifier(n_estimators=10, max_depth=10, max_features='sqrt', class_weight='balanced'),
                        RandomForestClassifier(n_estimators=10, max_depth=100, max_features='sqrt', class_weight='balanced')]

svm_models = [SVC(class_weight='balanced')]

naive_bayes_models = [GaussianNB()]

ada_boost_models = [AdaBoostClassifier(n_estimators=50), AdaBoostClassifier(n_estimators=100)]

histogram_boost_models = [HistGradientBoostingClassifier(), HistGradientBoostingClassifier(l2_regularization=1.0)]

gradient_boost_models = [GradientBoostingClassifier(), GradientBoostingClassifier(max_features='sqrt'),
                         GradientBoostingClassifier(max_features='log2')]

if __name__ == '__main__':
    rel_path = "Training_Images/"
    images_paths = get_paths_of_image_folders(rel_path)

    # Choose training set and test set
    training_paths = images_paths[2:]
    test_paths = images_paths[0:2]
    training_folder_names = [get_folder_name(path) for path in training_paths]
    test_folder_names = [get_folder_name(path) for path in test_paths]
    print("Train folders:", training_paths)
    print("Test folders:", test_paths)

    # Define data preprocessing options
    gray_scale = False
    normalize_hist = True
    with_image = False
    with_binary_patterns = False
    with_histograms = True
    with_segmentation = True
    with_mean = True
    with_std = True

    preprocessing_parameters = {'gray_scale': gray_scale, 'normalize_hist': normalize_hist, 'with_image': with_image,
                                'with_binary_patterns': with_binary_patterns, 'with_histograms': with_histograms,
                                'with_segmentation': with_segmentation, 'with_mean': with_mean, 'with_std': with_std}

    X_train, y_train = prepare_data_and_labels(training_paths, preprocessing_parameters)
    X_test, y_test = prepare_data_and_labels(test_paths, preprocessing_parameters)
    data_parameters = {'training_folder_names': training_folder_names, 'X_train': X_train, 'y_train': y_train,
                       'test_folder_names': test_folder_names, 'X_test': X_test, 'y_test': y_test}

    models = {'log_reg': log_reg_models, 'sgd': sgd_models, 'ridge_class': ridge_class_models,
              'decision_tree': decision_tree_models, 'random_forest': random_forest_models, 'svm': svm_models,
              'naive_bayes': naive_bayes_models, 'ada_boost': ada_boost_models,
              'histogram_boost': histogram_boost_models, 'gradient_boost': gradient_boost_models}

    for key, value in models.items():
        if key not in ['random_forest']:
            print(f"Skipped {key} models.")
            continue
        train_and_evaluate_modelgroup(modelgroup=value, modelgroup_name=key, data_params=data_parameters,
                                      preproc_params=preprocessing_parameters)
        print(f"Trained {key} models.")

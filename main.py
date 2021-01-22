from model_handling import *
from data_handling import *
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, LogisticRegressionCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# Define some different models for classification
log_reg_models = [LogisticRegression(penalty='none', max_iter=200, class_weight='balanced'),
                  LogisticRegression(penalty='l2', C=0.5, max_iter=200, class_weight='balanced'),
                  LogisticRegression(penalty='l1', C=0.5, max_iter=200, solver='saga', class_weight='balanced'),
                  LogisticRegression(penalty='elasticnet', C=0.5, solver='saga', l1_ratio=0.1, class_weight='balanced'),
                  LogisticRegression(penalty='l2', C=0.1, max_iter=200, class_weight='balanced'),
                  LogisticRegression(penalty='l1', C=0.1, max_iter=200, solver='saga', class_weight='balanced'),
                  LogisticRegression(penalty='elasticnet', C=0.1, solver='saga', l1_ratio=0.1, class_weight='balanced'),
                  LogisticRegression(penalty='l2', C=0.01, max_iter=200, class_weight='balanced'),
                  LogisticRegression(penalty='l1', C=0.01, max_iter=200, solver='saga', class_weight='balanced'),
                  LogisticRegression(penalty='elasticnet', C=0.01, solver='saga', l1_ratio=0.1,
                                     class_weight='balanced'),
                  LogisticRegression(penalty='l2', C=0.001, max_iter=200, class_weight='balanced'),
                  LogisticRegression(penalty='l1', C=0.001, max_iter=200, solver='saga', class_weight='balanced'),
                  LogisticRegression(penalty='elasticnet', C=0.001, solver='saga', l1_ratio=0.1,
                                     class_weight='balanced')]

sgd_models = [SGDClassifier(penalty='l2', alpha=0.01, class_weight='balanced'),
              SGDClassifier(penalty='l2', alpha=0.8, class_weight='balanced'),
              SGDClassifier(penalty='l2', alpha=2.0, class_weight='balanced')]

ridge_class_models = [RidgeClassifier(alpha=10.0, normalize=True, class_weight='balanced'),
                      RidgeClassifier(alpha=50.0, normalize=True, class_weight='balanced'),
                      RidgeClassifier(alpha=100.0, normalize=True, class_weight='balanced')]

decision_tree_models = [DecisionTreeClassifier(max_depth=10, max_features='sqrt', class_weight='balanced'),
                        DecisionTreeClassifier(max_depth=100, max_features='sqrt', class_weight='balanced'),
                        DecisionTreeClassifier(max_features='sqrt', class_weight='balanced')]

random_forest_models = [
    RandomForestClassifier(n_estimators=20, max_depth=3, max_features='sqrt', class_weight='balanced'),
    RandomForestClassifier(n_estimators=20, max_depth=10, max_features='sqrt', class_weight='balanced'),
    RandomForestClassifier(n_estimators=20, max_depth=100, max_features='sqrt', class_weight='balanced')]

svm_models = [LinearSVC(penalty='l2', dual=False, C=1.0, class_weight='balanced'),
              LinearSVC(penalty='l2', dual=False, C=0.1, class_weight='balanced'),
              LinearSVC(penalty='l1', dual=False, C=1.0, class_weight='balanced'),
              LinearSVC(penalty='l1', dual=False, C=0.1, class_weight='balanced'),
              SVC(kernel='linear', class_weight='balanced'),
              SVC(class_weight='balanced')]

naive_bayes_models = [GaussianNB()]

ada_boost_models = [AdaBoostClassifier(n_estimators=50), AdaBoostClassifier(n_estimators=100)]

histogram_boost_models = [HistGradientBoostingClassifier(), HistGradientBoostingClassifier(l2_regularization=1.0)]

gradient_boost_models = [GradientBoostingClassifier(), GradientBoostingClassifier(max_features='sqrt'),
                         GradientBoostingClassifier(max_features='log2')]

log_reg_cv_models = [
    LogisticRegressionCV(Cs=[0.0001, 0.001, 0.01, 0.1, 1], max_iter=200, penalty='l2', class_weight='balanced')]

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
    with_segmentation = False
    with_pca = False
    remove_low_var = False
    with_normalize = True
    with_mean = False
    with_std = False

    preprocessing_parameters = {'gray_scale': gray_scale, 'normalize_hist': normalize_hist, 'with_image': with_image,
                                'with_binary_patterns': with_binary_patterns, 'with_histograms': with_histograms,
                                'with_segmentation': with_segmentation, 'with_pca': with_pca,
                                'remove_low_var': remove_low_var, 'with_normalize': with_normalize,
                                'with_mean': with_mean, 'with_std': with_std}

    X_train, y_train = prepare_data_and_labels(training_paths, preprocessing_parameters)
    X_test, y_test = prepare_data_and_labels(test_paths, preprocessing_parameters)
    X_train, X_test = normalize_remove_var(X_train, X_test, preproc_params=preprocessing_parameters)

    data_parameters = {'training_folder_names': training_folder_names, 'X_train': X_train, 'y_train': y_train,
                       'test_folder_names': test_folder_names, 'X_test': X_test, 'y_test': y_test}

    models = {'log_reg': log_reg_models, 'sgd': sgd_models, 'ridge_class': ridge_class_models,
              'decision_tree': decision_tree_models, 'random_forest': random_forest_models, 'svm': svm_models,
              'naive_bayes': naive_bayes_models, 'ada_boost': ada_boost_models,
              'histogram_boost': histogram_boost_models, 'gradient_boost': gradient_boost_models,
              'log_reg_cv': log_reg_cv_models}

    for key, value in models.items():
        if key in ['sgd', 'decision_tree', 'histogram_boost', 'gradient_boost', 'log_reg_cv']:
            print(f"Skipped {key} models.")
            continue
        print(f"Trained {key} models.")
        train_and_evaluate_modelgroup(modelgroup=value, modelgroup_name=key, data_params=data_parameters,
                                      preproc_params=preprocessing_parameters)

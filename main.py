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
              SVC(C=1.0, kernel='linear', class_weight='balanced'),
              SVC(C=0.1, class_weight='balanced'), SVC(class_weight='balanced')]

naive_bayes_models = [GaussianNB()]

ada_boost_models = [AdaBoostClassifier(n_estimators=50), AdaBoostClassifier(n_estimators=100)]

histogram_boost_models = [HistGradientBoostingClassifier(), HistGradientBoostingClassifier(l2_regularization=1.0)]

gradient_boost_models = [GradientBoostingClassifier(), GradientBoostingClassifier(max_features='sqrt'),
                         GradientBoostingClassifier(max_features='log2')]

log_reg_cv_models = [
    LogisticRegressionCV(Cs=[0.0001, 0.001, 0.01, 0.1, 1], max_iter=200, penalty='l2', class_weight='balanced')]

if __name__ == '__main__':
    rel_path = "Hand_Selection/"
    images_paths = get_paths_of_image_folders(rel_path)

    # Define data preprocessing options, all but three must have boolean value
    gray_scale = False
    normalize_hist = True
    with_image = False
    with_binary_patterns = False
    histogram_params = (3, 64)      # must be None or a tuple of two integers, which describes (nb_divisions, nb_bins)
    with_segmentation = False
    nb_components_pca = 100     # must be None or a integer, which defines number of components
    threshold_low_var = None   # mus be None or a float in [0.0, 1.0], which defines threshold for minimal variance
    with_mean = False
    with_std = False

    preprocessing_parameters = {'gray_scale': gray_scale, 'normalize_hist': normalize_hist, 'with_image': with_image,
                                'with_binary_patterns': with_binary_patterns, 'histogram_params': histogram_params,
                                'with_segmentation': with_segmentation, 'nb_components_pca': nb_components_pca,
                                'threshold_low_var': threshold_low_var, 'with_mean': with_mean, 'with_std': with_std}
    test_size = 0.2
    X_train, X_test, y_train, y_test = prepare_train_and_test_set(images_paths, test_size, preprocessing_parameters)

    data_parameters = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

    models = {'log_reg': log_reg_models, 'sgd': sgd_models, 'ridge_class': ridge_class_models,
              'decision_tree': decision_tree_models, 'random_forest': random_forest_models, 'svm': svm_models,
              'naive_bayes': naive_bayes_models, 'ada_boost': ada_boost_models,
              'histogram_boost': histogram_boost_models, 'gradient_boost': gradient_boost_models,
              'log_reg_cv': log_reg_cv_models}

    for key, value in models.items():
        if key in ['sgd', 'decision_tree', 'histogram_boost', 'gradient_boost']:
            print(f"Skipped {key} models.")
            continue
        train_and_evaluate_modelgroup(modelgroup=value, modelgroup_name=key, data_params=data_parameters,
                                      preproc_params=preprocessing_parameters)
        print(f"Trained {key} models.")

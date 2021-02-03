from data_handling import get_paths_of_image_folders, prepare_train_and_test_set, prepare_data_and_labels, preprocess_data
from model_handling import train_and_evaluate_modelgroup, define_models, read_models, test_model, get_feature_dims


def main(training_session, data_path):
    images_paths = get_paths_of_image_folders(data_path)

    gray_scale = False  # only use gray scale image
    normalize_hist = True  # normalize histogram of image
    with_image = True  # use image
    with_binary_patterns = False  # use local binary patterns of image
    histogram_params = (3, 32)  # must be None or a tuple of two integers, which describes (nb_divisions, nb_bins)
    with_segmentation = 10  # must be None or a integer; segment image using k-means in color space
    nb_components_pca = 1000  # must be None or a integer, which defines number of components
    threshold_low_var = None  # must be None or a float in [0.0, 1.0], which defines threshold for minimal variance
    with_mean = False  # data gets shifted such that mean is 0.0
    with_std = False  # data gets scaled such that std is 1.0

    test_size = 0.2  # fraction of test set; only relevant if models are trained

    preprocessing_parameters = {'gray_scale': gray_scale, 'normalize_hist': normalize_hist, 'with_image': with_image,
                                'with_binary_patterns': with_binary_patterns, 'histogram_params': histogram_params,
                                'with_segmentation': with_segmentation, 'nb_components_pca': nb_components_pca,
                                'threshold_low_var': threshold_low_var, 'with_mean': with_mean, 'with_std': with_std}

    if training_session:
        log_reg = True
        sgd = False
        ridge_class = True
        decision_tree = False
        random_forest = False
        svm = False
        naive_bayes = False
        ada_boost = False
        histogram_boost = False
        gradient_boost = False
        log_reg_cv = True
        model_selection = {'log_reg': log_reg, 'sgd': sgd, 'ridge_class': ridge_class, 'decision_tree': decision_tree,
                           'random_forest': random_forest, 'svm': svm, 'naive_bayes': naive_bayes,
                           'ada_boost': ada_boost,
                           'histogram_boost': histogram_boost, 'gradient_boost': gradient_boost,
                           'log_reg_cv': log_reg_cv}
        models = define_models(model_selection=model_selection)

        X_train, X_test, y_train, y_test = prepare_train_and_test_set(images_paths, preprocessing_parameters,
                                                                      test_size=test_size)
        data_parameters = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
        for key, value in models.items():
            train_and_evaluate_modelgroup(modelgroup=value, modelgroup_name=key, data_params=data_parameters,
                                          preproc_params=preprocessing_parameters)
            print(f"Trained {key} models.")

    else:
        trained_models = ['log_reg_2044', 'log_reg_2058', 'log_reg_2072', 'log_reg_2086', 'log_reg_2085',
                          'log_reg_2087', 'svm_932', 'svm_934', 'svm_920', 'svm_927', 'svm_913']
        models = read_models(model_list=trained_models)
        same_nb_features = True
        feature_dims = get_feature_dims(models)
        for feature_dim in feature_dims:
            if feature_dim != list(feature_dims.values())[0]:
                print('Models use different number of features; computation takes longer.')
                same_nb_features = False
                k = 0
                break

        data, labels = prepare_data_and_labels(images_paths, preprocessing_parameters)
        if same_nb_features:
            X_test = preprocess_data(data, preprocessing_parameters)
        for key, value in models.items():
            if not same_nb_features:
                preprocessing_parameters['nb_components_pca'] = feature_dims[key]
                X_test = preprocess_data(data, preprocessing_parameters)
                k += 1
            print(key)
            test_model(value, X_test, labels)
    return None


if __name__ == '__main__':
    training_session = True
    data_path = "Candidate_Images/Small_Dataset/"
    main(training_session, data_path)

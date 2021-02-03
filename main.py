from data_handling import get_paths_of_image_folders, feature_computation, preprocess_data, set_export_data_name, \
    export_data, read_images, read_data
from model_handling import train_and_evaluate_modelgroup, define_models, read_models, test_model, get_feature_dims
import numpy as np


def main(training_session, data_path):
    images_paths = get_paths_of_image_folders(data_path)

    gray_scale = False  # only use gray scale image
    normalize_hist = True  # normalize histogram of image
    with_image = False  # use image
    with_binary_patterns = False  # use local binary patterns of image
    histogram_params = (3, 64)  # must be None or a tuple of two integers, which describes (nb_divisions, nb_bins)
    nb_segments = 10  # must be None or a integer; segment image using k-means in color space
    threshold_low_var = None  # must be None or a float in [0.0, 1.0], which defines threshold for minimal variance
    nb_components_pca = 20  # must be None or a integer, which defines number of components
    batch_size_pca = 500  # must be an integer, should be >= nb_features (ideally larger) and <= nb_images
    with_mean = True  # data gets shifted such that mean is 0.0
    with_std = False  # data gets scaled such that std is 1.0
    test_size = 0.2  # fraction of test set; only relevant if models are trained

    img_read_parameters = {'gray_scale': gray_scale, 'normalize_hist': normalize_hist}
    preprocessing_parameters = {'with_image': with_image, 'with_binary_patterns': with_binary_patterns,
                                'histogram_params': histogram_params, 'nb_segments': nb_segments,
                                'threshold_low_var': threshold_low_var, 'nb_components_pca': nb_components_pca,
                                'batch_size_pca': batch_size_pca, 'with_mean': with_mean, 'with_std': with_std}

    images_list, labels_list = read_images(data_path, img_read_parameters)
    labels = np.array(labels_list)
    data = preprocess_data(images_list, preprocessing_parameters)
    data_name = set_export_data_name(preprocessing_parameters)
    export_data(data, labels, data_name)

    if training_session:
        log_reg = True
        sgd = False
        ridge_class = True
        decision_tree = False
        random_forest = True
        svm = True
        naive_bayes = True
        ada_boost = True
        histogram_boost = False
        gradient_boost = False
        log_reg_cv = True
        model_selection = {'log_reg': log_reg, 'sgd': sgd, 'ridge_class': ridge_class, 'decision_tree': decision_tree,
                           'random_forest': random_forest, 'svm': svm, 'naive_bayes': naive_bayes,
                           'ada_boost': ada_boost, 'histogram_boost': histogram_boost, 'gradient_boost': gradient_boost,
                           'log_reg_cv': log_reg_cv}
        models = define_models(model_selection)
        data_parameters = img_read_parameters.update(preprocessing_parameters)
        data_parameters['test_size'] = test_size

        for key, value in models.items():
            train_and_evaluate_modelgroup(modelgroup=value,
                                          modelgroup_name=key,
                                          data=data,
                                          labels=labels,
                                          data_params=data_parameters)
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

        data, labels = feature_computation(images_paths, preprocessing_parameters)
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
    training_session = False
    data_path = "Candidate_Images/Balanced_Dataset"
    main(training_session=training_session, data_path=data_path)

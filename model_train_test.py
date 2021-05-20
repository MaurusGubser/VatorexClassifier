import os
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score, precision_score, \
    recall_score, plot_confusion_matrix, classification_report, plot_precision_recall_curve
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, LogisticRegressionCV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, \
    HistGradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from data_reading_writing import read_data_and_labels
from data_handling import downsize_false_candidates


def export_model(model, model_name):
    if not os.path.exists('Models_Trained'):
        os.mkdir('Models_Trained')
    rel_file_path = 'Models_Trained/' + model_name + '.sav'
    pickle.dump(model, open(rel_file_path, 'wb'))
    print("Model saved in", rel_file_path)
    return None


def export_model_stats_json(model_dict, model_name, data_dict):
    if not os.path.exists('Model_Statistics'):
        os.mkdir('Model_Statistics')
    rel_file_path = 'Model_Statistics/' + model_name + '.json'
    del model_dict['model']
    model_dict['model_stats_train']['conf_matrix'] = [int(k) for k in
                                                      model_dict['model_stats_train']['conf_matrix'].flatten()]
    model_dict['model_stats_test']['conf_matrix'] = [int(k) for k in
                                                     model_dict['model_stats_test']['conf_matrix'].flatten()]
    dict_params = OrderedDict({})
    dict_params.update(model_dict)
    dict_params.update(data_dict)
    with open(rel_file_path, 'w') as outfile:
        json.dump(dict_params, outfile, indent=4)
    print("Model statistics saved in", rel_file_path)
    return None


def export_model_stats_csv(model_dict, model_name, data_dict):
    if not os.path.exists('Model_Statistics'):
        os.mkdir('Model_Statistics')
    filename = 'Model_Statistics/Model_Statistics.csv'
    if not os.path.exists(filename):
        title_string = 'Model name,Model_params,TRAIN Accuracy,Acc. Balanced,Precision,Recall,F1 Score,TEST Accuracy,Acc. Balanced,Precision,Recall,F1 Score,'
        for i in data_dict.keys():
            title_string = title_string + str(i) + ','
        title_string = title_string + '\n'
        with open(filename, 'w') as initfile:
            initfile.write(title_string)

    model_string = model_name + ',' + str(model_dict['model_params']).replace(',', '') + ','
    for key, model_value in model_dict['model_stats_train'].items():
        if key == 'conf_matrix':
            continue
        model_string = model_string + str(model_value) + ','
    for key, model_value in model_dict['model_stats_test'].items():
        if key == 'conf_matrix':
            continue
        model_string = model_string + str(model_value) + ','
    for data_value in data_dict.values():
        model_string = model_string + str(data_value).replace(',', '') + ','
    model_string = model_string + '\n'
    with open(filename, 'a') as outfile:
        outfile.write(model_string)

    print("Model statistics of {} appended to {}".format(model_name, filename))
    return None


def read_model_stats_json(stats_path):
    with open(stats_path) as infile:
        stats_dict = json.load(infile)
    stats_dict['model_stats_train']['conf_matrix'] = np.reshape(stats_dict['model_stats_train']['conf_matrix'], (2, 2))
    stats_dict['model_stats_test']['conf_matrix'] = np.reshape(stats_dict['model_stats_test']['conf_matrix'], (2, 2))
    return stats_dict


def train_model(model, X_train, y_train):
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    print('Training time: {:.0f}min {:.0f}s'.format((end_time - start_time) / 60, (end_time - start_time) % 60))
    return model


def evaluate_model(model, X, y, paths):
    start_time = time.time()
    y_pred = model.predict(X)
    end_time = time.time()
    print('Evaluating time: {:.0f}min {:.0f}s'.format((end_time - start_time) / 60, (end_time - start_time) % 60))
    stats_dict = OrderedDict([('conf_matrix', confusion_matrix(y, y_pred)), ('acc', accuracy_score(y, y_pred)),
                              ('acc_balanced', balanced_accuracy_score(y, y_pred)),
                              ('prec', precision_score(y, y_pred)), ('rcll', recall_score(y, y_pred)),
                              ('f1_scr', f1_score(y, y_pred))])
    misclassified_imgs, true_pos_imgs = list_fp_fn_tp_images(y, y_pred, paths)
    return stats_dict, misclassified_imgs, true_pos_imgs


def evaluate_trained_model(path_test_data, data_params, path_trained_model, model_name):
    model = pickle.load(open(path_trained_model, 'rb'))
    X_test, y_test, paths_images = read_data_and_labels(path_test_data, data_params)
    y_pred = model.predict(X_test)
    misclassified_imgs, true_pos_imgs = list_fp_fn_tp_images(y_test, y_pred, paths_images)
    export_evaluation_images_model(misclassified_imgs, true_pos_imgs, model_name, 'Test')
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rcll = recall_score(y_test, y_pred)
    print('F1 score: {}, Precision: {}, Recall: {}'.format(f1, prec, rcll))
    plot_confusion_matrix(model, X_test, y_test)
    return None


def list_fp_fn_tp_images(y_true, y_pred, paths_images):
    misclassified_imgs = paths_images[y_true + y_pred == 1]
    correct_imgs = paths_images[y_true + y_pred == 2]
    return misclassified_imgs, correct_imgs


def export_evaluation_images_model(misclassified_images, true_pos_images, model_name, train_test):
    model_dir = 'Evaluation_Images/' + model_name
    if not os.path.exists('Evaluation_Images'):
        os.mkdir('Evaluation_Images')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    dir_misclassified = model_dir + '/' + train_test + '_Misclassified/'
    export_images(misclassified_images, dir_misclassified)
    dir_true_pos = model_dir + '/' + train_test + '_TruePos/'
    export_images(true_pos_images, dir_true_pos)
    print('Exported evaluation images to {}'.format(model_dir))
    return None


def export_images(images_list, export_directory):
    if not os.path.exists(export_directory):
        os.mkdir(export_directory)
    filename_list = export_directory + 'image_list.txt'
    np.savetxt(filename_list, np.sort(images_list), delimiter=' ', fmt='%s')
    for path in images_list:
        path = path.replace('(', '\(')
        path = path.replace(')', '\)')
        os.system('cp {} ./{}'.format(path, export_directory))
    return None


def get_name_index(model_name, folder_name, file_format):
    idx = 0
    if os.path.exists(folder_name):
        list_model_paths = [str(path) for path in Path(folder_name).rglob(model_name + '*.' + file_format)]
        idx = len(list_model_paths)
    return idx


def train_and_test_modelgroup(modelgroup, modelgroup_name, X_train, X_test, y_train, y_test, paths_train, paths_test,
                              data_params):
    index = get_name_index(modelgroup_name, 'Model_Statistics/', 'json')

    dict_data = OrderedDict([('training_size', y_train.size), ('training_nb_mites', int(np.sum(y_train))),
                             ('test_size', y_test.size), ('test_nb_mites', int(np.sum(y_test))),
                             ('feature_size', X_train.shape[1])])
    dict_data.update(data_params)

    for i in range(0, len(modelgroup)):
        model_name = modelgroup_name + '_' + str(index + i)
        dict_model = OrderedDict([('model', modelgroup[i]), ('model_params', modelgroup[i].get_params())])

        dict_model['model'] = train_model(dict_model['model'], X_train, y_train)
        dict_model['model_stats_train'], misclassified_train, true_pos_train = evaluate_model(dict_model['model'],
                                                                                              X_train, y_train,
                                                                                              paths_train)
        dict_model['model_stats_test'], misclassified_test, true_pos_test = evaluate_model(dict_model['model'], X_test,
                                                                                           y_test, paths_test)

        # export_model(dict_model['model'], model_name)
        # export_model_stats_json(dict_model, model_name, dict_data)
        export_model_stats_csv(dict_model, model_name, dict_data)
        export_evaluation_images_model(misclassified_train, true_pos_train, model_name, 'Train')
        export_evaluation_images_model(misclassified_test, true_pos_test, model_name, 'Test')

    return None


def train_and_test_model_selection(model_selection, folder_path, data_params, test_size):
    data, labels, paths_images = read_data_and_labels(folder_path, data_params)
    models = define_models(model_selection)

    data, labels, paths_images = downsize_false_candidates(data, labels, paths_images, data_params['percentage_true'])
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(data,
                                                                                 labels,
                                                                                 paths_images,
                                                                                 test_size=test_size,
                                                                                 random_state=42,
                                                                                 stratify=labels)
    del data

    for key, value in models.items():
        train_and_test_modelgroup(value, key, X_train, X_test, y_train, y_test, paths_train, paths_test, data_params)
    return None


def define_models(model_selection):
    log_reg_models = [LogisticRegression(penalty='none', max_iter=200, class_weight='balanced'),
                      LogisticRegression(penalty='l2', C=1.0, max_iter=200, class_weight='balanced'),
                      LogisticRegression(penalty='l1', C=1.0, max_iter=200, solver='saga', class_weight='balanced'),
                      LogisticRegression(penalty='elasticnet', C=1.0, solver='saga', l1_ratio=0.1,
                                         class_weight='balanced'),
                      LogisticRegression(penalty='l2', C=0.1, max_iter=200, class_weight='balanced'),
                      LogisticRegression(penalty='l1', C=0.1, max_iter=200, solver='saga', class_weight='balanced'),
                      LogisticRegression(penalty='elasticnet', C=0.1, solver='saga', l1_ratio=0.1,
                                         class_weight='balanced'),
                      LogisticRegression(penalty='l2', C=0.01, max_iter=200, class_weight='balanced'),
                      LogisticRegression(penalty='l1', C=0.01, max_iter=200, solver='saga', class_weight='balanced'),
                      LogisticRegression(penalty='elasticnet', C=0.01, solver='saga', l1_ratio=0.1,
                                         class_weight='balanced'),
                      LogisticRegression(penalty='l2', C=0.001, max_iter=200, class_weight='balanced'),
                      LogisticRegression(penalty='l1', C=0.001, max_iter=200, solver='saga', class_weight='balanced'),
                      LogisticRegression(penalty='elasticnet', C=0.001, solver='saga', l1_ratio=0.1,
                                         class_weight='balanced')]

    sgd_models = [SGDClassifier(penalty='l2', alpha=0.01, class_weight='balanced'),
                  SGDClassifier(penalty='l2', alpha=0.5, class_weight='balanced'),
                  SGDClassifier(penalty='l2', alpha=2.0, class_weight='balanced')]

    ridge_class_models = [RidgeClassifier(alpha=1.0, normalize=True, max_iter=None, class_weight='balanced'),
                          RidgeClassifier(alpha=10.0, normalize=True, max_iter=None, class_weight='balanced'),
                          RidgeClassifier(alpha=50.0, normalize=True, max_iter=None, class_weight='balanced'),
                          RidgeClassifier(alpha=100.0, normalize=True, max_iter=None, class_weight='balanced')]

    decision_tree_models = [DecisionTreeClassifier(max_depth=10, max_features='sqrt', class_weight='balanced'),
                            DecisionTreeClassifier(max_depth=100, max_features='sqrt', class_weight='balanced'),
                            DecisionTreeClassifier(max_features='sqrt', class_weight='balanced')]

    random_forest_models = [
        RandomForestClassifier(n_estimators=20, max_depth=3, max_features='sqrt', class_weight='balanced'),
        RandomForestClassifier(n_estimators=20, max_depth=10, max_features='sqrt', class_weight='balanced'),
        RandomForestClassifier(n_estimators=20, max_depth=100, max_features='sqrt', class_weight='balanced'),
        RandomForestClassifier(n_estimators=50, max_depth=3, max_features='sqrt', class_weight='balanced'),
        RandomForestClassifier(n_estimators=50, max_depth=10, max_features='sqrt', class_weight='balanced'),
        RandomForestClassifier(n_estimators=50, max_depth=100, max_features='sqrt', class_weight='balanced'),
        RandomForestClassifier(n_estimators=100, max_depth=3, max_features='sqrt', class_weight='balanced'),
        RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt', class_weight='balanced'),
        RandomForestClassifier(n_estimators=100, max_depth=100, max_features='sqrt', class_weight='balanced')]

    l_svm_models = [LinearSVC(penalty='l2', dual=False, C=1.0, class_weight='balanced', max_iter=500),
                    LinearSVC(penalty='l2', dual=False, C=0.1, class_weight='balanced', max_iter=500),
                    LinearSVC(penalty='l1', dual=False, C=1.0, class_weight='balanced', max_iter=500),
                    LinearSVC(penalty='l1', dual=False, C=0.1, class_weight='balanced', max_iter=500)]

    nl_svm_models = [SVC(C=0.1, class_weight='balanced'),
                     SVC(C=1.0, class_weight='balanced'),
                     SVC(C=5.0, class_weight='balanced'),
                     SVC(C=0.1, kernel='poly', class_weight='balanced'),
                     SVC(C=0.1, kernel='poly', class_weight='balanced'),
                     SVC(C=5.0, kernel='poly', class_weight='balanced')]

    naive_bayes_models = [GaussianNB()]

    ada_boost_models = [AdaBoostClassifier(n_estimators=50),
                        AdaBoostClassifier(n_estimators=100),
                        AdaBoostClassifier(n_estimators=200),
                        AdaBoostClassifier(n_estimators=500),
                        AdaBoostClassifier(n_estimators=50, learning_rate=0.1),
                        AdaBoostClassifier(n_estimators=100, learning_rate=0.1),
                        AdaBoostClassifier(n_estimators=200, learning_rate=0.1),
                        AdaBoostClassifier(n_estimators=500, learning_rate=0.1)]

    histogram_boost_models = [HistGradientBoostingClassifier(max_iter=10),
                              HistGradientBoostingClassifier(max_iter=100, l2_regularization=0.1),
                              HistGradientBoostingClassifier(max_iter=100, l2_regularization=1.0),
                              HistGradientBoostingClassifier(max_iter=100, l2_regularization=5.0)]

    gradient_boost_models = [GradientBoostingClassifier(n_estimators=100),
                             GradientBoostingClassifier(n_estimators=100, max_features='sqrt'),
                             GradientBoostingClassifier(n_estimators=100, max_features='log2'),
                             GradientBoostingClassifier(n_estimators=200),
                             GradientBoostingClassifier(n_estimators=200, max_features='sqrt'),
                             GradientBoostingClassifier(n_estimators=200, max_features='log2'),
                             GradientBoostingClassifier(n_estimators=500),
                             GradientBoostingClassifier(n_estimators=500, max_features='sqrt'),
                             GradientBoostingClassifier(n_estimators=500, max_features='log2')]

    log_reg_cv_models = [
        LogisticRegressionCV(Cs=[0.0001, 0.001, 0.01, 0.1, 1], max_iter=200, penalty='l2', class_weight='balanced')]

    estimators = [[('svc', SVC(C=1.0, class_weight='balanced')),
                   ('hist_boost', HistGradientBoostingClassifier(max_iter=300, l2_regularization=5.0))],
                  [('nb', GaussianNB()),
                   ('hist_boost', HistGradientBoostingClassifier(max_iter=300, l2_regularization=5.0))],
                  [('ridge', RidgeClassifier(alpha=1.0, normalize=True, max_iter=None, class_weight='balanced')),
                   ('hist_boost', HistGradientBoostingClassifier(max_iter=300, l2_regularization=5.0))],
                  [('log_reg', LogisticRegression(penalty='elasticnet', C=0.1, solver='saga', l1_ratio=0.1,
                                                  class_weight='balanced')),
                   ('hist_boost', HistGradientBoostingClassifier(max_iter=300, l2_regularization=5.0))]]

    stacked_models = [StackingClassifier(estimators=estimators[0]), StackingClassifier(estimators=estimators[1]),
                      StackingClassifier(estimators=estimators[2]), StackingClassifier(estimators=estimators[3])]

    experimental_models = [HistGradientBoostingClassifier(max_iter=300),
                           HistGradientBoostingClassifier(max_iter=300, l2_regularization=0.1),
                           HistGradientBoostingClassifier(max_iter=300, l2_regularization=1.0),
                           HistGradientBoostingClassifier(max_iter=300, l2_regularization=5.0),
                           HistGradientBoostingClassifier(max_iter=300, l2_regularization=10.0),
                           HistGradientBoostingClassifier(max_iter=300),
                           HistGradientBoostingClassifier(max_iter=300, l2_regularization=0.1, max_depth=3),
                           HistGradientBoostingClassifier(max_iter=300, l2_regularization=1.0, max_depth=3),
                           HistGradientBoostingClassifier(max_iter=300, l2_regularization=5.0, max_depth=3),
                           HistGradientBoostingClassifier(max_iter=300, l2_regularization=10.0, max_depth=3),
                           HistGradientBoostingClassifier(max_iter=300),
                           HistGradientBoostingClassifier(max_iter=500, l2_regularization=0.1),
                           HistGradientBoostingClassifier(max_iter=500, l2_regularization=1.0),
                           HistGradientBoostingClassifier(max_iter=500, l2_regularization=5.0),
                           HistGradientBoostingClassifier(max_iter=500, l2_regularization=10.0)]

    models = OrderedDict([('log_reg', log_reg_models), ('sgd', sgd_models), ('ridge_class', ridge_class_models),
                          ('decision_tree', decision_tree_models), ('random_forest', random_forest_models),
                          ('l_svm', l_svm_models), ('nl_svm', nl_svm_models), ('naive_bayes', naive_bayes_models),
                          ('ada_boost', ada_boost_models), ('histogram_boost', histogram_boost_models),
                          ('gradient_boost', gradient_boost_models), ('log_reg_cv', log_reg_cv_models),
                          ('stacked', stacked_models), ('experimental', experimental_models)])

    for key, value in model_selection.items():
        if not value:
            models.pop(key)

    return models


def read_models(model_list):
    model_dict = OrderedDict({})
    for name in model_list:
        model = pickle.load(open('Models_Trained/' + name + '.sav', 'rb'))
        model_dict[name] = model
    return model_dict


def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Non-mite', 'Mite']))
    plot_confusion_matrix(model, X_test, y_test, display_labels=['Non-mite', 'Mite'])
    plot_precision_recall_curve(model, X_test, y_test)
    plt.show()
    return None


def get_feature_dims(model_dict):
    feature_dims = OrderedDict({})
    for key, value in model_dict.items():
        model_type = key[0:key.rfind('_')]
        if model_type in ['log_reg', 'sgd', 'ridge_class', 'log_reg_cv']:
            feature_dims[key] = value.coef_.shape[1]
        elif model_type in ['svm']:
            feature_dims[key] = value.support_vectors_.shape[1]
        elif model_type in ['naive_bayes']:
            feature_dims[key] = value.theta_.shape[1]
        elif model_type in ['ada_boost', 'gradient_boost']:
            feature_dims[key] = value.feature_importance_.shape[0]
        elif model_type in ['histogram_boost']:
            feature_dims[key] = value.is_categorical_.shape[0]
        else:
            feature_dims[key] = value.n_features_
    return feature_dims

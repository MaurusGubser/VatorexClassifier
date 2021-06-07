import numpy as np
from collections import OrderedDict
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC

from model_parameter_tuning import cross_validate_model, grid_search_model, plot_learning_curve_model
from model_train_test import train_and_test_model_selection
from sequential_model import train_and_test_sequential_models, define_sequential_models


# ----- data parameters -----
read_image = False  # True or False
read_hist = 'context'    # 'candidate', 'context' or False
with_image = False  # use image
with_binary_patterns = False  # use local binary patterns of image
histogram_params = (3, 16)  # must be None or a tuple of two integers, which describes (nb_divisions, nb_bins)
nb_segments = 6  # must be None or a integer; segment image using k-means in color space
threshold_low_var = None  # must be None or a float in [0.0, 1.0], which defines threshold for minimal variance
nb_components_pca = 100  # must be None or a integer, which defines number of components
batch_size_pca = 1000  # must be an integer, should be >= nb_features (ideally larger) and <= nb_images
hist_hsl = True
hist_h = True
hist_s = True
hist_l = True
percentage_true = 0.05  # desired percentage of trues in data set
with_mean = False  # data gets shifted such that mean is 0.0
with_std = False  # data gets scaled such that std is 1.0

data_parameters = OrderedDict([('read_image', read_image), ('read_hist', read_hist), ('with_image', with_image),
                               ('with_binary_patterns', with_binary_patterns), ('histogram_params', histogram_params),
                               ('nb_segments', nb_segments), ('threshold_low_var', threshold_low_var),
                               ('nb_components_pca', nb_components_pca), ('batch_size_pca', batch_size_pca),
                               ('hist_hsl', hist_hsl), ('hist_h', hist_h), ('hist_s', hist_s), ('hist_l', hist_l),
                               ('percentage_true', percentage_true), ('with_mean', with_mean), ('with_std', with_std)])
test_size = 0.2  # fraction of test set

# ----- train and evaluate models -----
train_models = False

log_reg = False
sgd = False
ridge_class = False
decision_tree = False
random_forest = False
l_svm = True
nl_svm = False
naive_bayes = False
ada_boost = False
histogram_boost = False
gradient_boost = False
log_reg_cv = False
stacked = False
experimental = False
model_selection = OrderedDict([('log_reg', log_reg), ('sgd', sgd), ('ridge_class', ridge_class),
                               ('decision_tree', decision_tree), ('random_forest', random_forest),
                               ('l_svm', l_svm), ('nl_svm', nl_svm), ('naive_bayes', naive_bayes),
                               ('ada_boost', ada_boost), ('histogram_boost', histogram_boost),
                               ('gradient_boost', gradient_boost), ('log_reg_cv', log_reg_cv),
                               ('stacked', stacked), ('experimental', experimental)])

# ----- train and evaluate sequential models -----
evaluate_sequential = False

names_sequential = ['svc_hist', 'nb_hist', 'ridge_hist', 'logreg_hist', 'rf_hist']

models_recall = [SVC(C=1.0, class_weight='balanced'), GaussianNB(),
                 RidgeClassifier(alpha=1.0, normalize=True, max_iter=None, class_weight='balanced'),
                 LogisticRegression(penalty='none', C=0.1, solver='lbfgs', l1_ratio=0.1, class_weight='balanced'),
                 RandomForestClassifier(n_estimators=20, max_depth=3, max_features='sqrt', class_weight='balanced')]

models_precision = [HistGradientBoostingClassifier(max_iter=300, l2_regularization=5.0, max_depth=3),
                    HistGradientBoostingClassifier(max_iter=300, l2_regularization=5.0, max_depth=3),
                    HistGradientBoostingClassifier(max_iter=300, l2_regularization=5.0, max_depth=3),
                    HistGradientBoostingClassifier(max_iter=300, l2_regularization=5.0, max_depth=3),
                    HistGradientBoostingClassifier(max_iter=300, l2_regularization=5.0, max_depth=3)]

# ----- cross-validation for one parameter -----
cross_validation = False

model_cv = RandomForestClassifier()
model_name = 'RandomForest'
model_parameter = 'n_estimators'  # e.g. learning_rate, max_iter, max_depth, l2_regularization, max_bins depending on model
semilog = True  # if x axis should be logarithmic
# parameter_range = np.array([0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]) # learning_rate
# parameter_range = np.array([5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200])#, 300, 400, 500, 600, 700, 800, 900, 1000])    # max_iter
# parameter_range = np.array([2, 3, 5, 7, 9, 15, 20, 25, 30, 50, 100, 200])   # max_depth
# parameter_range = np.insert(np.logspace(-2, 3, 15), 0, 0.0)    # l2_regularization
# parameter_range = np.array([2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 255])   # max_bins
parameter_range = np.array([1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 100, 150, 200])  # n_estimators
# parameter_range = np.array([0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])    # learning_rate
# parameter_range = np.array([0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 5.0, 10.0, 100.0, 1000.0])   # C, alpha
nb_split_cv = 10  # number of split cvs

cv_parameters = OrderedDict([('model_name', model_name), ('model_parameter', model_parameter),
                             ('parameter_range', parameter_range), ('semilog', semilog),
                             ('nb_split_cv', nb_split_cv)])

# ----- grid search for several parameters -----
grid_search = True

model_gs = HistGradientBoostingClassifier()
model_name = 'Histogram_boost'
scoring_parameters = ['recall', 'precision', 'f1']

learning_rate = ('learning_rate', np.array([0.1, 0.15, 0.2, 0.25]))
max_iter = ('max_iter', np.array([100]))
max_depth = ('max_depth', np.array([20]))
l2_regularization = ('l2_regularization', np.insert(np.logspace(-2, 2, 6), 0, 0.0))
max_bins = ('max_bins', np.array([31]))

parameters_grid = OrderedDict([learning_rate, max_iter, max_depth, l2_regularization, max_bins])
nb_split_cv = 10    # number of split cvs
gs_parameters = OrderedDict([('model_name', model_name), ('parameters_grid', parameters_grid), ('scoring_parameters', scoring_parameters), ('nb_split_cv', nb_split_cv)])

# ----- evaluate trained model ------
evaluate_model = False
path_trained_model = 'path/to/trained/model'
path_test_data = 'path/to/test/data/folders'
model_name = 'model_name_for_export'

if __name__ == '__main__':
    path_image_folders = "Candidate_Images/Mite4_Dataset_contextellipsis/"
    if train_models + evaluate_sequential + cross_validation + grid_search > 1:
        print('Only one of evaluate_models, evaluate_sequential, cross_validation, grid_search should be True.')
    elif train_models:
        train_and_test_model_selection(model_selection, path_image_folders, data_parameters, test_size)
    elif evaluate_sequential:
        models_sequential = define_sequential_models(names_sequential, models_recall, models_precision)
        train_and_test_sequential_models(models_sequential, path_image_folders, data_parameters, test_size)
    elif cross_validation:
        cross_validate_model(model_cv, path_image_folders, data_parameters, cv_parameters)
    elif grid_search:
        grid_search_model(model_gs, path_image_folders, data_parameters, gs_parameters, test_size)
    elif evaluate_model:
        evaluate_trained_model(path_test_data, data_parameters, path_trained_model, model_name)
    else:
        print('No option chosen.')

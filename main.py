import numpy as np
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from lightgbm import LGBMClassifier

from lgbm_train_export import export_Vatorex_model
from model_parameter_tuning import cross_validation_one_param, grid_search_multi_param
from model_train_test import compare_different_models, evaluate_trained_model
from roc_precrcll_curves import plot_roc_precrcll_curves

# ----- data parameters -----
read_image = False  # True or False
read_hist = 'context'  # must be 'candidate', 'context' or None
with_image = None  # must be None or a scalar, which defines downsize factor; use image
with_binary_patterns = False  # use local binary patterns of image
histogram_params = None  # (3, 16)  # must be None or a tuple of two integers, which describes (nb_divisions, nb_bins)
nb_segments = 20  # must be None or an integer; segment image using k-means in color space
threshold_low_var = None  # must be None or a float in [0.0, 1.0], which defines threshold for minimal variance
nb_components_pca = None  # must be None or an integer, which defines number of components
batch_size_pca = None  # must be an integer, should be >= nb_features (ideally larger) and <= nb_images
hist_hsl = True
hist_h = True
hist_s = True
hist_l = True
quadratic_features = False  # use basis 1, x_i, x_i**2, no mixed terms
with_mean = False  # data gets shifted such that mean is 0.0
with_std = False  # data gets scaled such that std is 1.0
with_false1 = False  # use false1 labelled data

data_parameters = OrderedDict([('read_image', read_image), ('read_hist', read_hist), ('with_image', with_image),
                               ('with_binary_patterns', with_binary_patterns), ('histogram_params', histogram_params),
                               ('nb_segments', nb_segments), ('threshold_low_var', threshold_low_var),
                               ('nb_components_pca', nb_components_pca), ('batch_size_pca', batch_size_pca),
                               ('hist_hsl', hist_hsl), ('hist_h', hist_h), ('hist_s', hist_s), ('hist_l', hist_l),
                               ('quadratic_features', quadratic_features), ('with_mean', with_mean),
                               ('with_std', with_std), ('with_false1', with_false1)])
test_size = 0.10  # must be float in (0,1); fraction of test set

path_image_folders = 'Candidate_Images/Series_matching05_mindist015_train/'  # 'Candidate_Images/Small_Series/'

# ----- train and compare models -----
compare_models = False

log_reg = True
sgd = False
ridge_class = False
decision_tree = False
random_forest = True
l_svm = True
nl_svm = False
naive_bayes = True
ada_boost = True
histogram_boost = True
gradient_boost = False

model_selection = OrderedDict([('log_reg', log_reg), ('sgd', sgd), ('ridge_class', ridge_class),
                               ('decision_tree', decision_tree), ('random_forest', random_forest),
                               ('l_svm', l_svm), ('nl_svm', nl_svm), ('naive_bayes', naive_bayes),
                               ('ada_boost', ada_boost), ('histogram_boost', histogram_boost),
                               ('gradient_boost', gradient_boost)])

use_class_weight = None  # give classes weight according to their size; either 'balanced' or None
reweight_posterior = False  # if posterior probabilities should be reweighted for prediction

# ----- cross-validation for one parameter -----
cross_validation = True

model_cv = LGBMClassifier(is_unbalanced='balanced')
model_name = 'LGBM_balanced'
model_parameter = 'n_estimators'  # e.g. learning_rate, max_iter, max_depth, l2_regularization, max_bins depending on model
semilog = False  # if x-axis should be logarithmic
# parameter_range = np.array([0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]) # learning_rate
parameter_range = np.array([100, 200, 300, 400, 500, 600])  # max_iter/n_estimators
# parameter_range = np.array([2, 3, 5, 7, 9, 15, 20, 25, 30, 50, 100, 200])   # max_depth
# parameter_range = np.array([2, 3, 5, 7, 9, 15, 20])   # max_leaf_nodes/num_leaves
# parameter_range = np.insert(np.logspace(-2, 3, 10), 0, 0.0)  # l2_regularization/reg_lambda
# parameter_range = np.array([2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 255])   # max_bins
# parameter_range = np.insert(np.logspace(-2, 4, 5), 0, 0.0)    # C, alpha
nb_split_gs = 10  # number of split cvs

cv_parameters = OrderedDict([('model_name', model_name), ('model_parameter', model_parameter),
                             ('parameter_range', parameter_range), ('semilog', semilog),
                             ('nb_split_cv', nb_split_gs)])

# ----- grid search for several parameters -----
grid_search = False

model_gs = LGBMClassifier(is_unbalanced='balanced')
model_name = 'LGBM_balanced'
scoring_parameters = ['recall', 'precision', 'f1', 'roc_auc']
refit_param = 'f1'

max_bins = ('max_bins', np.array([15, 63, 255]))
learning_rate = ('learning_rate', np.array([0.1, 0.2, 0.3]))
n_estimators = ('n_estimators', np.array([50, 100, 300]))
max_depth = ('max_depth', np.array([4, 20, 50, -1]))
num_leaves = ('num_leaves', np.array([3, 7, 15, 31]))
reg_lambda = ('reg_lambda', np.insert(np.logspace(-2, 2, 5), 0, 0.0))
class_weight = ('class_weight', [None, 'balanced'])

parameters_grid = OrderedDict([max_bins, learning_rate, n_estimators, max_depth, num_leaves, reg_lambda, class_weight])
nb_split_cv = 10  # number of split cvs
gs_parameters = OrderedDict([('model_name', model_name), ('parameters_grid', parameters_grid),
                             ('scoring_parameters', scoring_parameters), ('refit_param', refit_param),
                             ('nb_split_cv', nb_split_cv)])

# ----------- plot curves ----------------------
plot_curves = False
clf = LGBMClassifier()

# ----- evaluate trained model ------
evaluate_model = False
path_trained_model = 'path/to/trained/model.txt'
path_test_data = 'path/to/data/'
model_name = 'data_model_name'

# ----- train and export Vatorex model ------
train_export_Vatorex = False

cv = 10
parameters_lgbm = {'objective': 'binary',
                   'num_iterations': 100,
                   'learning_rate': 0.1,
                   'force_col_wise': True,
                   'n_jobs': -1,
                   'reg_lambda': 1.0,
                   'num_leaves': 31,  # std 31
                   'max_depth': -1,  # std -1
                   'is_unbalance': False}

# ----- apply parameters and code ------
if __name__ == '__main__':
    if compare_models + cross_validation + grid_search + evaluate_model + train_export_Vatorex > 1:
        raise AssertionError('Only one of evaluate_models, cross_validation, grid_search should be True.')
    elif compare_models:
        if test_size is None:
            raise ValueError('Parameter test_size cannot be None.')
        compare_different_models(model_selection, path_image_folders, data_parameters, test_size,
                                 use_class_weight, reweight_posterior)
    elif cross_validation:
        cross_validation_one_param(model_cv, path_image_folders, data_parameters, cv_parameters)
    elif grid_search:
        grid_search_multi_param(model_gs, path_image_folders, data_parameters, gs_parameters, test_size,
                                reweight_posterior)
    elif plot_curves:
        plot_roc_precrcll_curves(clf, path_image_folders, data_parameters, test_size)
    elif evaluate_model:
        evaluate_trained_model(path_test_data, data_parameters, path_trained_model, model_name)
    elif train_export_Vatorex:
        export_Vatorex_model(path_image_folders, data_parameters, test_size, cv, parameters_lgbm)
    else:
        print('No option chosen.')

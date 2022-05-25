import numpy as np
from collections import OrderedDict
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
# histogram over RGB space, rectangular area
histogram_params = None  # (3, 16)  # must be None or a tuple of two integers, which describes (nb_divisions, nb_bins)
# histogram over HSL, ellipse area
hist_hsl = True  # use joint histograms over HSL color space
hist_h = True  # use H histograms over HSL color space
hist_s = True  # use S histograms over HSL color space
hist_l = True  # use L histograms over HSL color space
with_mean = False  # data gets shifted such that mean is 0.0
with_std = False  # data gets scaled such that std is 1.0
with_false1 = False  # use false1 labelled data

data_parameters = OrderedDict([('read_image', read_image), ('read_hist', read_hist), ('with_image', with_image),
                               ('with_binary_patterns', with_binary_patterns), ('histogram_params', histogram_params),
                               ('hist_hsl', hist_hsl), ('hist_h', hist_h), ('hist_s', hist_s), ('hist_l', hist_l),
                               ('with_mean', with_mean), ('with_std', with_std), ('with_false1', with_false1)])
test_size = 0.10  # must be float in (0,1); fraction of test set
path_image_folders = 'path/to/data'  # path to data

# ----- train and compare models -----
compare_models = True

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

# ----- cross-validation for one parameter -----
cross_validation = False

model_cv = LGBMClassifier(is_unbalanced='balanced')
model_name = 'LGBM_balanced'

model_parameter = 'n_estimators'  # e.g. C, learning_rate, max_iter, max_depth, l2_regularization depending on model
parameter_range = np.array([100, 200, 300, 400, 500])
semilog = False  # if x-axis should be logarithmic
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

# some LGBM parameters; any set of parameters of LGBM can be used
n_estimators = ('n_estimators', np.array([50, 100, 300]))
max_depth = ('max_depth', np.array([4, 20, 50, -1]))
num_leaves = ('num_leaves', np.array([3, 7, 15, 31]))
reg_lambda = ('reg_lambda', np.insert(np.logspace(-2, 2, 5), 0, 0.0))

parameters_grid = OrderedDict([n_estimators, max_depth, num_leaves, reg_lambda])
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
                   'reg_lambda': 1.0,
                   'max_depth': -1,
                   'is_unbalance': False,
                   'n_jobs': -1}    # use all cores

# ----- apply parameters and code ------
if __name__ == '__main__':
    if compare_models + cross_validation + grid_search + evaluate_model + train_export_Vatorex > 1:
        raise AssertionError('Only one of evaluate_models, cross_validation, grid_search should be True.')
    elif compare_models:
        if test_size is None:
            raise ValueError('Parameter test_size cannot be None.')
        compare_different_models(model_selection, path_image_folders, data_parameters, test_size)
    elif cross_validation:
        cross_validation_one_param(model_cv, path_image_folders, data_parameters, cv_parameters)
    elif grid_search:
        grid_search_multi_param(model_gs, path_image_folders, data_parameters, gs_parameters, test_size)
    elif plot_curves:
        plot_roc_precrcll_curves(clf, path_image_folders, data_parameters, test_size)
    elif evaluate_model:
        evaluate_trained_model(path_test_data, data_parameters, path_trained_model, model_name)
    elif train_export_Vatorex:
        export_Vatorex_model(path_image_folders, data_parameters, test_size, cv, parameters_lgbm)
    else:
        print('No option chosen.')

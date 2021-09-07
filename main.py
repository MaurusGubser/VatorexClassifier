import numpy as np
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from lightgbm import LGBMClassifier

from lgbm_train_export import export_GUI_model
from model_parameter_tuning import cross_validate_model, grid_search_model
from model_train_test import train_and_test_model_selection, evaluate_trained_model

# ----- data parameters -----
read_image = False  # True or False
read_hist = 'context'  # must be 'candidate', 'context' or False
with_image = None  # must be None or a scalar, which defines downsize factor; use image
with_binary_patterns = False  # use local binary patterns of image
histogram_params = None  # (3, 16)  # must be None or a tuple of two integers, which describes (nb_divisions, nb_bins)
nb_segments = None  # must be None or a integer; segment image using k-means in color space
threshold_low_var = None  # must be None or a float in [0.0, 1.0], which defines threshold for minimal variance
nb_components_pca = None  # must be None or a integer, which defines number of components
batch_size_pca = None  # must be an integer, should be >= nb_features (ideally larger) and <= nb_images
hist_hsl = True
hist_h = True
hist_s = True
hist_l = True
quadratic_features = False  # use basis 1, x_i, x_i**2, no mixed terms
with_mean = False  # data gets shifted such that mean is 0.0
with_std = False  # data gets scaled such that std is 1.0

data_parameters = OrderedDict([('read_image', read_image), ('read_hist', read_hist), ('with_image', with_image),
                               ('with_binary_patterns', with_binary_patterns), ('histogram_params', histogram_params),
                               ('nb_segments', nb_segments), ('threshold_low_var', threshold_low_var),
                               ('nb_components_pca', nb_components_pca), ('batch_size_pca', batch_size_pca),
                               ('hist_hsl', hist_hsl), ('hist_h', hist_h), ('hist_s', hist_s), ('hist_l', hist_l),
                               ('quadratic_features', quadratic_features), ('with_mean', with_mean),
                               ('with_std', with_std)])
test_size = 0.10  # must be float in (0,1); fraction of test set
undersampling_rate = 0.05  # must be None or float in [0,1]; false candidates get undersampled to according ratio
oversampling_rate = 0.30  # must be None or float in [0,1]; true candidates get oversample to according ratio

# ----- train and evaluate models -----
train_models = True

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
handicraft = False

model_selection = OrderedDict([('log_reg', log_reg), ('sgd', sgd), ('ridge_class', ridge_class),
                               ('decision_tree', decision_tree), ('random_forest', random_forest),
                               ('l_svm', l_svm), ('nl_svm', nl_svm), ('naive_bayes', naive_bayes),
                               ('ada_boost', ada_boost), ('histogram_boost', histogram_boost),
                               ('gradient_boost', gradient_boost), ('handicraft', handicraft)])

use_weights = None  # weights for model fitting; must be None, 'balanced' or [weight_0, weight_1] in percent

# ----- cross-validation for one parameter -----
cross_validation = False

model_cv = LGBMClassifier(n_estimators=10, class_weight='balanced')
model_name = 'LGBMClassifier'
model_parameter = 'reg_lambda'  # e.g. learning_rate, max_iter, max_depth, l2_regularization, max_bins depending on model
semilog = True  # if x axis should be logarithmic
# parameter_range = np.array([0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]) # learning_rate
# parameter_range = np.array([5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200])#, 300, 400, 500, 600, 700, 800, 900, 1000])    # max_iter/n_estimators
# parameter_range = np.array([2, 3, 5, 7, 9, 15, 20, 25, 30, 50, 100, 200])   # max_depth
# parameter_range = np.array([2, 3, 5, 7, 9, 15, 20])   # max_leaf_nodes/num_leaves
parameter_range = np.insert(np.logspace(-2, 3, 15), 0, 0.0)  # l2_regularization/reg_lambda
# parameter_range = np.array([2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 255])   # max_bins
# parameter_range = np.array([1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 100, 150, 200])  # n_estimators
# parameter_range = np.array([0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])    # learning_rate
# parameter_range = np.array([0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 5.0, 10.0, 100.0, 1000.0])   # C, alpha
nb_split_cv = 5  # number of split cvs

cv_parameters = OrderedDict([('model_name', model_name), ('model_parameter', model_parameter),
                             ('parameter_range', parameter_range), ('semilog', semilog),
                             ('nb_split_cv', nb_split_cv)])

# ----- grid search for several parameters -----
grid_search = False

model_gs = LinearSVC()
model_name = 'LinearSVC'
scoring_parameters = ['recall', 'precision', 'f1']
refit_param = 'f1'

Cs = ('C', np.insert(np.logspace(-2, 3, 12), 0, 0.0))
class_weight = ('class_weight', [None, 'balanced'])
max_iter = ('max_iter', np.array([-1, 10, 50, 100, 500]))

parameters_grid = OrderedDict([Cs, class_weight, max_iter])
nb_split_cv = 3  # number of split cvs
gs_parameters = OrderedDict([('model_name', model_name), ('parameters_grid', parameters_grid),
                             ('scoring_parameters', scoring_parameters), ('refit_param', refit_param),
                             ('nb_split_cv', nb_split_cv)])

# ----- evaluate trained model ------
evaluate_model = False
path_trained_model = '/home/maurus/PyCharm_Projects/Vatorex_Classifier/Models_Trained/LinearSVC_8.sav'
path_test_data = '/home/maurus/PyCharm_Projects/Vatorex_Classifier/Candidate_Images/Mite4_relabelledtol02/200328-S09(labeled)/'
model_name = 'LinearSVC_1_200812R09AS'

# ----- train and export model for GUI ------
train_export_GUI = False
path_data = 'GUI_Model_Export/Model_true02_jointmarginal/Mite4_relabelledtol02_False_context_None_False_None_None_None_None_None_True_True_True_True_0.2_False_False_False.npz'
undersampling_GUI = 0.2  # must be None or float in (0, 1)
oversampling_GUI = None     # must be None or float in (0, 1)
cv = 5
parameters_lgbm = {'task': 'train',
                   'objective': 'binary',
                   'boosting': 'gbdt',
                   'num_iterations': 30,
                   'learning_rate': 0.2,
                   'num_leaves': 31,
                   'num_threads': 4,
                   'deterministic': True,
                   'max_depth': 20,
                   'lambda_l2': 50.0,
                   'max_bin': 31,
                   'n_iter_no_change': 10,
                   'is_unbalance': True,
                   'metric': 'binary_logloss'}
export_name = 'LightGBM_Model_02true_joint_balanced.txt'

# ----- apply parameters and code ------
if __name__ == '__main__':
    path_image_folders = "Candidate_Images/Mite4_relabelledtol02/200328-S09(labeled)/"
    if train_models + cross_validation + grid_search + evaluate_model + train_export_GUI > 1:
        raise AssertionError('Only one of evaluate_models, cross_validation, grid_search should be True.')
    elif train_models:
        if test_size is None:
            raise ValueError('Parameter test_size cannot be None.')
        train_and_test_model_selection(model_selection, path_image_folders, data_parameters, test_size, undersampling_rate, oversampling_rate, use_weights)
    elif cross_validation:
        cross_validate_model(model_cv, path_image_folders, data_parameters, cv_parameters, undersampling_rate, oversampling_rate, use_weights)
    elif grid_search:
        grid_search_model(model_gs, path_image_folders, data_parameters, gs_parameters, test_size, undersampling_rate, oversampling_rate, use_weights)
    elif evaluate_model:
        evaluate_trained_model(path_test_data, data_parameters, path_trained_model, model_name)
    elif train_export_GUI:
        export_GUI_model(path_data, undersampling_GUI, oversampling_GUI, cv, parameters_lgbm, export_name)
    else:
        print('No option chosen.')

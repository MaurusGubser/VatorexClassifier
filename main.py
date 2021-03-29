import numpy as np
from collections import OrderedDict
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from model_parameter_tuning import cross_validate_model
from model_train_test import train_and_test_model_selection
from sequential_model import train_and_test_sequential_models, define_sequential_models

# ----- data parameters -----
read_image = False
read_hist = True
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

# ----- training models -----
evaluate_models = False

log_reg = False
sgd = False
ridge_class = False
decision_tree = False
random_forest = False
l_svm = False
nl_svm = False
naive_bayes = False
ada_boost = True
histogram_boost = True
gradient_boost = True
log_reg_cv = False
stacked = False
experimental = False
model_selection = OrderedDict([('log_reg', log_reg), ('sgd', sgd), ('ridge_class', ridge_class),
                               ('decision_tree', decision_tree), ('random_forest', random_forest),
                               ('l_svm', l_svm), ('nl_svm', nl_svm), ('naive_bayes', naive_bayes),
                               ('ada_boost', ada_boost), ('histogram_boost', histogram_boost),
                               ('gradient_boost', gradient_boost), ('log_reg_cv', log_reg_cv),
                               ('stacked', stacked), ('experimental', experimental)])

# ----- sequential models -----
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

# ----- cross-validation models -----
cross_validation = True

model = HistGradientBoostingClassifier(max_iter=100)
model_parameter = 'l2_regularization'  # e.g. max_iter, max_bins, depending on model
semilog = True  # if x axis should be logarithmic
parameter_range = np.logspace(-4, 1, 10)    # l2_regularization
# parameter_range = np.array([5, 10, 15, 20, 30, 40, 50, 75, 100, 200, 300])    # max_iter
# parameter_range = np.array([2, 3, 7, 15, 31, 63, 127, 255])   # max_bins
scoring_parameter = 'f1'    # f1, recall, precision
nb_split_cv = 5   # number of split cvs
cv_parameters = OrderedDict([('model_parameter', model_parameter), ('parameter_range', parameter_range),
                             ('semilog', semilog), ('scoring_parameter', scoring_parameter),
                             ('nb_split_cv', nb_split_cv)])

if __name__ == '__main__':
    folder_path = "Candidate_Images/Mite4_Dataset_renderellipsis_test/"
    if evaluate_models:
        train_and_test_model_selection(model_selection, folder_path, data_parameters, test_size)
    elif evaluate_sequential:
        models_sequential = define_sequential_models(names_sequential, models_recall, models_precision)
        train_and_test_sequential_models(models_sequential, folder_path, data_parameters, test_size)
    elif cross_validation:
        cross_validate_model(model, folder_path, data_parameters, cv_parameters)

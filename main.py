from model_train_test import read_models, train_and_test_model_selection

# ----- data parameters -----
read_image = False
read_hist = True
with_image = False  # use image
with_binary_patterns = False  # use local binary patterns of image
histogram_params = (3, 32)  # must be None or a tuple of two integers, which describes (nb_divisions, nb_bins)
nb_segments = 10  # must be None or a integer; segment image using k-means in color space
threshold_low_var = None  # must be None or a float in [0.0, 1.0], which defines threshold for minimal variance
nb_components_pca = 100  # must be None or a integer, which defines number of components
batch_size_pca = 1000  # must be an integer, should be >= nb_features (ideally larger) and <= nb_images
hist_hsl = True
hist_h = True
hist_s = True
hist_l = True
ratio_true_false = 0.05  # desired true-false ratio of data set
with_mean = False  # data gets shifted such that mean is 0.0
with_std = False  # data gets scaled such that std is 1.0

data_parameters = {'read_image': read_image, 'read_hist': read_hist, 'with_image': with_image,
                   'with_binary_patterns': with_binary_patterns, 'histogram_params': histogram_params,
                   'nb_segments': nb_segments, 'threshold_low_var': threshold_low_var,
                   'nb_components_pca': nb_components_pca, 'batch_size_pca': batch_size_pca, 'hist_hsl': hist_hsl,
                   'hist_h': hist_h, 'hist_s': hist_s, 'hist_l': hist_l, 'ratio_true_false': ratio_true_false,
                   'with_mean': with_mean, 'with_std': with_std}
test_size = 0.2  # fraction of test set

# ----- training models -----
log_reg = False
sgd = False
ridge_class = False
decision_tree = False
random_forest = False
svm = False
naive_bayes = False
ada_boost = False
histogram_boost = True
gradient_boost = True
log_reg_cv = False
model_selection = {'log_reg': log_reg, 'sgd': sgd, 'ridge_class': ridge_class, 'decision_tree': decision_tree,
                   'random_forest': random_forest, 'svm': svm, 'naive_bayes': naive_bayes,
                   'ada_boost': ada_boost, 'histogram_boost': histogram_boost, 'gradient_boost': gradient_boost,
                   'log_reg_cv': log_reg_cv}
"""
trained_models = ['log_reg_2044', 'log_reg_2058', 'log_reg_2072', 'log_reg_2086', 'log_reg_2085', 'log_reg_2087',
                  'svm_932', 'svm_934', 'svm_920', 'svm_927', 'svm_913']
models = read_models(model_list=trained_models)
"""

if __name__ == '__main__':
    folder_path = "Candidate_Images/Mite4_Dataset_Cleaned/"
    train_and_test_model_selection(model_selection, folder_path, data_parameters, test_size)

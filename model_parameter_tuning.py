from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, cross_val_score, cross_validate, validation_curve
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, \
    HistGradientBoostingClassifier

from data_reading_writing import load_data_and_labels, concatenate_data
from data_handling import downsize_false_candidates


read_image = False
read_hist = True
path_data = 'Preprocessed_Data/Mite4_Dataset_Cleaned_False_True_False_False_(3, 32)_10_None_100_1000_True_True_True_True_0.05_False_False.npz'
data_images, data_histograms, labels = load_data_and_labels(path_data)
data = concatenate_data(data_images, data_histograms, read_image, read_hist)
data, labels = downsize_false_candidates(data, labels, 0.05)

nl_svm_models = [SVC(C=0.1, class_weight='balanced'),
                 SVC(C=1.0, class_weight='balanced'),
                 SVC(C=5.0, class_weight='balanced'),
                 SVC(C=0.1, kernel='poly', class_weight='balanced'),
                 SVC(C=0.1, kernel='poly', class_weight='balanced'),
                 SVC(C=5.0, kernel='poly', class_weight='balanced')]

histogram_boost_models = [HistGradientBoostingClassifier(max_iter=100),
                          HistGradientBoostingClassifier(max_iter=100, l2_regularization=0.1),
                          HistGradientBoostingClassifier(max_iter=100, l2_regularization=1.0),
                          HistGradientBoostingClassifier(max_iter=100, l2_regularization=5.0)]

models = histogram_boost_models
score = ['balanced_accuracy', 'recall', 'precision', 'f1']


def cross_validation_model(model, data, labels, score):
    cv = 10
    return cross_validate(model, X=data, y=labels, scoring=score, cv=cv, return_train_score=True)


cv_eval = []
for model in models:
    cv_eval_model = cross_validation_model(model, data, labels, score)
    cv_eval.append(cv_eval_model)
    print(cv_eval_model)

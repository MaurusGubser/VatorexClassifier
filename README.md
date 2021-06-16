# Vatorex_Classifier
Classifier to detect varroa mites in a sample of candidate images.

The training data (images and histograms) should be stored in a subfolder "Training_Images". In the file data_reading_writing.py are functions defined for reading and exporting data; in the file data_handling.py are functions defined for preprocessing and rearranging the data; in the file model_train_test.py are functions defined for specifying, training, evaluating and exporting models and its statistics on the data; in the file sequential_model.py are functions defined for testing a sequential model; in the file model_parameter_tuning.py are functions defined for cross-validation and parameter tuning.

To use the main.py file, one should choose settings for the data and choose between model evaluating, sequential model evaluating, cross-validations mode or grid search mode using the corresponding boolean variables. If the data was already used with given parameters, it is load from "Preprocessed_Data"; if not, it is stored in a folder "Preprocessed_Data" after reading and preprocessing.

*Train models*
If a model is trained and evaluated, its statistics are stored in one json file (one model) and appended to one csv file "Model_Statistics.csv" (all models) in the folder "Model_Statistics". The model itself is stored in the folder "Models_Trained". The true positive and misclassified images (false positive, false negative) are stored in a subfolder of "Evaluation_Images".

*Train sequential models*
One has to define a sequence of models: one for recall, one for precision. Its statistics are then stored in one json file (one model) and appended to one csv file "Sequential_Model_Statistics.csv" (all models) in the folder "Sequential_Model_Statistics".

*Cross validation*
One has to define a model, a model parameter to be tune, the range of corresponding parameter and a scoring metric, for which the model should be evaluated. The precision, recall and F1 score are computed for the different parameters; corresponding graphs are stored in "CV_Plots". No model is stored.

*Grid search*
One has to define a model, a list of parameters and a range for each parameter. Then, a grid search over those parameter ranges is performed. The statistics of the 10 best performing models are stored in "GridSearch_Statistics". The true positive and misclassified images (false positive, false negative) are stored in a subfolder of "Evaluation_Images".

*Evaluate model*
One has to define the path to the test data, the path to a pre-trained model and a model name, which is used for the export. The F1, precision and recall scores are computed and saved in a json-file, and the true positive and misclassified images (false positive, false negative) are stored in a subfolder "Evaluation_Images".

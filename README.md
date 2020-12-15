# Vatorex_Classifier
Classifier to detect varroa mites in a sample of candidate images.

The training images are stored in the folder "Training_Images". In the file data_handling.py are functions defined for reading and preprocessing the data; in the file model_handling.py are functions defined for training, evaluating and exporting models and its statistics on the data.

To use the main.py file, one has to define one or several models using scikit-learn, e.g. LogisticRegression models and to define a collection of the provided data set as training and test set. The models are then trained and evaluated. The models itself get exported in the folder "Models_Trained", the statistics of the model on the test data get exported in one json file per model and in one csv file "Model_Statistics.csv" for all models in the folder "Model_Statistics".

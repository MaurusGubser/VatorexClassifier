# Vatorex_Classifier
Classifier to detect varroa mites in a sample of candidate images.

The training data (images and histograms) should be stored in a subfolder "Training_Images". In the file data_reading_writing.py are functions defined for reading and exporting data; in the file data_handling.py are functions defined for preprocessing and rearranging the data; in the file model_train_test.py are functions defined for specifying, training, evaluating and exporting models and its statistics on the data.

If data is read and preprocessed, it is stored in a folder "Preprocessed_Data"; if a model is trained and evaluated, its statistics are stored in one json file (one model) and appended to one csv file "Model_Statistics.csv" (all models) in the folder "Model_Statistics". The model itself is stored in the folder "Models_Trained".

To use the main.py file, one should choose settings for the data and choose, which group of models should be trained. If one wants to change the parameters of one model separately, one has to do this in the model_train_test.py file.

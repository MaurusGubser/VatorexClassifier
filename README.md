# Vatorex_Classifier
Classifier to detect varroa mites in a sample of candidate images.

The script contains different functions with different purposes:

- compare_different_models: comparing different classifiers
- cross_validation_one_param: make cross validation for a single hyper parameter
- grid_search_multi_param: make a grid search for finding the best hyper parameters for a model
- plot_roc_precrcll_curves: plot the ROC curve and precision-recall-curve for on model
- evaluate_trained_model: evaluate a trained model on test data
- export_Vatorex_model: train and export a LightGBM model for classification

Each function can be executed by setting the corresponding boolean value to True and then executing the main.py script.


### Data
The data should be stored in a subfolder "Candidate_Images" and the images and histograms need to be stored in subfolders "extracted". In the file data_reading_writing.py, functions for reading and exporting data are defined; in the file data_handling.py, functions for preprocessing and rearranging the data are defined. If data is read and preprocessed, it is stored in a folder "Preprocessed_Data".
In the main.py file, one can set several parameters to define the exact nature of the data, e.g. to use images or histograms, to normalise the data,...

### Train and compare models
Different models can be chosen for comparison.

### Cross validation
For a chosen model, on has to set a (meaningful) name, a model parameter, a range for this parameter and the number of splits for cross validation.

### Grid search
For a chosen model, on has to set a (meaningful) name, tuples consisting of a model parameter and the range for this parameter and the number of splits for cross validation.

### Plot curves
Set a model.

### Train and export 
Set number of splits for cross validation and the parameters for the LightGBM model.
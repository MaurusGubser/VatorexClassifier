import os.path
import numpy as np
import lightgbm as lgb

from data_reading_writing import reload_data_and_labels


# ---------------------- working dir ----------------------
working_dir = 'LGBM_Comparison/220405_Test1/'
assert os.path.exists(working_dir), 'Directory {} does not exist.'.format(working_dir)

# ---------------------- load data and labels ----------------------
path_data = working_dir + 'data.npz'    # must be already stored as .npz, for convenience
_, data_hist, labels, _ = reload_data_and_labels(path_data)
print('Data of shape {}'.format(data_hist.shape))

# ---------------------- load model ----------------------
path_model = working_dir + 'model.txt'
model = lgb.Booster(model_file=path_model)

# ---------------------- model prediction ----------------------
pred_proba = model.predict(data_hist)
pred_labels = np.around(pred_proba)

# ---------------------- save data and prediction ----------------------

export_path_data = working_dir + 'data.txt'
with open(export_path_data, 'w') as out_data:
    np.savetxt(out_data, data_hist, delimiter=',')

export_path_labels = working_dir + 'labels.txt'
with open(export_path_labels, 'w') as out_labels:
    np.savetxt(out_labels, labels)

export_path_predictions = working_dir + 'pred_proba.txt'
with open(export_path_predictions, 'w') as out_pred_proba:
    np.savetxt(out_pred_proba, pred_proba)

export_path_pred_labels = working_dir + 'pred_labels.txt'
with open(export_path_pred_labels, 'w') as out_pred_labels:
    np.savetxt(out_pred_labels, pred_labels)

print('Saved data, labels and predictions.')

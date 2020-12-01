import os
from pathlib import Path
import numpy as np
import re
import cv2
from sklearn.preprocessing import scale


def read_data_from_folder(path_image_folder, normalize_hist=True):
    image_folder = Path(path_image_folder).rglob('*.jpg')
    images_paths = [str(path) for path in image_folder]
    nb_images = len(images_paths)
    data = []
    labels = []

    pattern_true = r'true'
    pattern_false = r'false'
    for i in range(0, nb_images):
        image = cv2.imread(images_paths[i])
        if normalize_hist:
            image = histogram_equalization(image)
        data.append(image.flatten())
        if re.search(pattern_true, images_paths[i]) is not None:
            labels.append(1)
        elif re.search(pattern_false, images_paths[i]) is not None:
            labels.append(0)
        else:
            print(images_paths[i])
            raise AssertionError("Image path does not contain 'true' or 'false'.")
    labels = np.array(labels)
    data = np.array(data)
    return data, labels


def normalize_data(data, with_mean=True, with_std=True):
    return scale(data, axis=-1, with_mean=with_mean, with_std=with_std)


def histogram_equalization(image):
    if image.ndim == 2:
        return cv2.equalizeHist(image)
    elif image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
        image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
        return image
    else:
        print('Image shape:', image.shape)
        raise AssertionError('Image has neither one channel (gray scale) nor three channels (RGB).')


def read_data(folder_list, normalize_mean=True, normalize_std=True, normalize_hist=True):
    data, labels = read_data_from_folder(folder_list[0])
    for i in range(1, len(folder_list)):
        X, y = read_data_from_folder(folder_list[i], normalize_hist=normalize_hist)
        data = np.concatenate((data, X))
        labels = np.concatenate((labels, y))
    data = normalize_data(data, with_mean=normalize_mean, with_std=normalize_std)
    return data, labels


def get_paths_image_folders(path_folder):
    images_paths = []
    for root, dirs, files in os.walk(path_folder):
        for name in dirs:
            images_paths.append(path_folder + name)
    return images_paths


def get_folder_name(path_folder):
    name_start = path_folder.rfind('/')
    folder_name = path_folder[name_start+1:]
    return folder_name

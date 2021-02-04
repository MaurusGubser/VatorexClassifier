import os
from pathlib import Path
import re
import numpy as np
from skimage.io import imread
from skimage.exposure import equalize_adapthist

from data_handling import preprocess_data


def read_images_from_folder(path_folder, gray_scale, normalize_hist):
    image_folder = Path(path_folder).rglob('*.jpg')
    images_paths = [str(path) for path in image_folder]
    images = []
    labels = []
    pattern_true = r'true'
    pattern_false = r'false'

    for path in images_paths:
        image = imread(path, as_gray=gray_scale)
        if normalize_hist:
            image = equalize_adapthist(image)
        images.append(image)
        if re.search(pattern_true, path):
            labels.append(1)
        elif re.search(pattern_false, path):
            labels.append(0)
        else:
            raise AssertionError(f"Image path is {path} and does not contain 'true' or 'false'.")
    return images, labels


def read_images(folder_list):
    images = []
    labels = []
    for folder_path in folder_list:
        imgs, lbls = read_images_from_folder(folder_path, gray_scale=False, normalize_hist=True)
        images = images + imgs
        labels = labels + lbls
    return images, labels


def get_paths_of_image_folders(path_folder):
    folder_list = []
    for root, dirs, files in os.walk(path_folder):
        for name in dirs:
            folder_list.append(path_folder + name)
    return folder_list


def get_folder_name(path_folder):
    name_start = path_folder[:len(path_folder) - 1].rfind('/')
    folder_name = path_folder[name_start + 1:]
    folder_name = folder_name.replace('/', '')
    return folder_name


def set_export_data_name(folder_name, preprocessing_params):
    name = folder_name
    for value in preprocessing_params.values():
        name = name + '_' + str(value)
    return name


def export_data(data, labels, data_name):
    path_data = 'Preprocessed_Data/' + data_name + '_data'
    path_labels = 'Preprocessed_Data/' + data_name + '_labels'
    if os.path.exists(path_data + '.npy'):
        print('Preprocessed data with these parameters already exported.')
        return None
    if not os.path.exists('Preprocessed_Data'):
        os.mkdir('Preprocessed_Data')
    np.save(path_data, data, allow_pickle=False)
    np.save(path_labels, labels, allow_pickle=False)
    print(f'Saved data and labels in files {path_data} and {path_labels}')
    return None


def load_data_and_labels(path_data):
    data = np.load(path_data)
    path_labels = path_data.replace('data', 'labels')
    labels = np.load(path_labels)
    return data, labels


def read_data_and_labels(path, preprocessing_params):
    folder_name = get_folder_name(path)
    path_preprocessed = 'Preprocessed_Data/' + set_export_data_name(folder_name, preprocessing_params) + '_data.npy'
    if os.path.exists(path_preprocessed):
        data, labels = load_data_and_labels(path_preprocessed)
        print(f'Re-loaded preprocessed data and labels from {path_preprocessed}')
        return data, labels
    else:
        folder_list = get_paths_of_image_folders(path)
        images_list, labels_list = read_images(folder_list)
        labels = np.array(labels_list)
        data = preprocess_data(images_list, preprocessing_params)
        data_name = set_export_data_name(folder_name, preprocessing_params)
        export_data(data, labels, data_name)
        return data, labels

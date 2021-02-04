import os
from pathlib import Path
import numpy as np
import re
from skimage.io import imread
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, IncrementalPCA
from skimage.color import rgb2ycbcr, ycbcr2rgb
from skimage.feature import local_binary_pattern
from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity
from skimage.segmentation import slic

import time


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
            image = equalize_histogram_adaptive(image)
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


def equalize_histogram(image):
    image = rescale_intensity(image, out_range=(0.0, 1.0))
    if image.ndim == 2:
        return equalize_hist(image)
    elif image.ndim == 3 and image.shape[-1] == 3:
        image = rgb2ycbcr(image)
        image[:, :, 0] = rescale_intensity(equalize_hist(image[:, :, 0]), out_range=(16.0, 235.0))
        image = ycbcr2rgb(image)
        image = rescale_intensity(image, out_range=(0.0, 1.0))
        return image
    else:
        print('Image shape:', image.shape)
        raise ValueError(
            f'Image has neither one channel (gray scale) nor three channels (RGB); got shape {image.shape} instead.')


def equalize_histogram_adaptive(image):
    return equalize_adapthist(image)


def compute_local_binary_pattern(image, nb_pts=None, radius=3):
    if image.ndim == 2:
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))
    if nb_pts is None:
        nb_pts = 8 * radius
    image_lbp = np.zeros(image.shape)
    for ch in range(0, image.shape[-1]):
        image_lbp[:, :, ch] = local_binary_pattern(image[:, :, ch], nb_pts, radius)
    return image_lbp


def compute_histograms(image, nb_divisions, nb_bins):
    width = image.shape[0]
    length = image.shape[1]
    if image.ndim == 2:
        image = np.reshape(image, (width, length, 1))
    width_subregion = width // nb_divisions
    length_subregion = length // nb_divisions
    histograms = np.zeros((image.shape[-1], nb_divisions ** 2, nb_bins))
    for ch in range(0, image.shape[-1]):
        for i in range(0, nb_divisions):
            for j in range(0, nb_divisions):
                sub_img = image[i * width_subregion:(i + 1) * width_subregion,
                          j * length_subregion:(j + 1) * length_subregion, ch]
                histograms[ch, i * nb_divisions + j, :] = np.histogram(sub_img, bins=nb_bins)[0]
    return histograms


def segment_image(img, nb_segments):
    return slic(img, n_segments=nb_segments)


def scale_data(data, with_mean, with_std):
    return scale(data, axis=-1, with_mean=with_mean, with_std=with_std)


def feature_computation(images_list, with_image, with_binary_patterns, histogram_params, nb_segments):
    start = time.time()
    data = []
    for img in images_list:
        data_img = np.empty(0)
        if with_image:
            data_img = np.append(data_img, img.flatten())
        if with_binary_patterns:
            data_img = np.append(data_img, compute_local_binary_pattern(img).flatten())
        if histogram_params:
            nb_divisions, nb_bins = histogram_params
            data_img = np.append(data_img,
                                 compute_histograms(img, nb_divisions=nb_divisions, nb_bins=nb_bins).flatten())
        if nb_segments:
            data_img = np.append(data_img, segment_image(img, nb_segments).flatten())
        data.append(data_img)

    data = np.array(data)
    end = time.time()
    print(f"Computed features in {(end - start) / 60:.1f} minutes; data of shape {data.shape}")
    return data


def dimension_reduction(data, threshold_low_var, nb_components_pca, batch_size_pca):
    start = time.time()
    old_shape = data.shape
    if threshold_low_var:
        selector = VarianceThreshold(threshold=threshold_low_var)
        data = selector.fit_transform(data)
    if nb_components_pca:
        pca = IncrementalPCA(n_components=nb_components_pca, batch_size=batch_size_pca)
        # data = normalize(data)
        pca.fit(data)
        data = pca.transform(data)
    end = time.time()
    print(f"Dimensionality reduction took {(end - start) / 60:.1f} minutes; reduction from {old_shape} to {data.shape}")
    return data


def preprocess_data(images_list, preprocessing_params):
    with_image = preprocessing_params['with_image']
    with_binary_patterns = preprocessing_params['with_binary_patterns']
    histogram_params = preprocessing_params['histogram_params']
    nb_segments = preprocessing_params['nb_segments']
    threshold_low_var = preprocessing_params['threshold_low_var']
    nb_components_pca = preprocessing_params['nb_components_pca']
    batch_size_pca = preprocessing_params['batch_size_pca']
    with_mean = preprocessing_params['with_mean']
    with_std = preprocessing_params['with_std']

    if not (with_image or with_binary_patterns or histogram_params or nb_segments):
        raise ValueError(
            "At least one of 'with_image', 'with_binary_patterns', 'histogram_params', 'nb_segments' has to be True.")
    data = feature_computation(images_list, with_image, with_binary_patterns, histogram_params, nb_segments)
    if threshold_low_var or nb_components_pca or batch_size_pca:
        data = dimension_reduction(data, threshold_low_var, nb_components_pca, batch_size_pca)
    if with_mean or with_std:
        data = scale_data(data, with_mean, with_std)
    return data


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

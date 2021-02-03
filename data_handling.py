import os
from pathlib import Path
import numpy as np
import re
from skimage.io import imread, imread_collection
from sklearn.preprocessing import scale, normalize
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, IncrementalPCA
from skimage.color import rgb2ycbcr, ycbcr2rgb
from skimage.feature import local_binary_pattern
from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity
from skimage.segmentation import slic

import time


def read_images_from_folder(path_folder, gray_scale=False, hist_eq=True):
    image_folder = Path(path_folder).rglob('*.jpg')
    images_paths = [str(path) for path in image_folder]
    images = []
    labels = []
    pattern_true = r'true'
    pattern_false = r'false'

    for path in images_paths:
        image = imread(path, as_gray=gray_scale)
        if hist_eq:
            image = equalize_histogram_adaptive(image)
        images.append(image)
        if re.search(pattern_true, path):
            labels.append(1)
        elif re.search(pattern_false, path):
            labels.append(0)
        else:
            raise AssertionError(f"Image path is {path} and does not contain 'true' or 'false'.")
    return images, labels


def read_images(folder_list, gray_scale=False, hist_eq=True):
    images = []
    labels = []
    for folder_path in folder_list:
        imgs, lbls = read_images_from_folder(folder_path, gray_scale=gray_scale, hist_eq=hist_eq)
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


def prepare_data_and_labels(folder_list, preproc_params):
    start = time.time()
    images, labels = read_images(folder_list, preproc_params['gray_scale'], preproc_params['normalize_hist'])
    data = []
    if not (preproc_params['with_image'] or preproc_params['with_binary_patterns'] or preproc_params['histogram_params']
            or preproc_params['with_segmentation']):
        raise ValueError("At least one of 'with_image', 'with_binary_patterns', 'histogram_params', 'with_segmentation' has to be True.")
    for img in images:
        data_img = np.empty(0)
        if preproc_params['with_image']:
            data_img = np.append(data_img, img.flatten())
        if preproc_params['with_binary_patterns']:
            data_img = np.append(data_img, compute_local_binary_pattern(img).flatten())
        if preproc_params['histogram_params'] is not None:
            nb_divisions, nb_bins = preproc_params['histogram_params']
            data_img = np.append(data_img, compute_histograms(img, nb_divisions=nb_divisions, nb_bins=nb_bins))
        if preproc_params['with_segmentation']:
            nb_segments = preproc_params['with_segmentation']
            data_img = np.append(data_img, segment_image(img, nb_segments).flatten())
        data.append(data_img)

    data = np.array(data, dtype='float32')
    labels = np.array(labels)
    end = time.time()
    print(f"Read data and labels in {(end - start) / 60:.1f} minutes; data of shape {data.shape}")
    return data, labels


def preprocess_data(data, preproc_params):
    start = time.time()
    if preproc_params['nb_components_pca'] is not None:
        #pca = PCA(n_components=preproc_params['nb_components_pca'])
        pca = IncrementalPCA(n_components=preproc_params['nb_components_pca'], batch_size=1000)
        data = normalize(data)
        pca.fit(data)
        data = pca.transform(data)
    if preproc_params['with_mean'] or preproc_params['with_std']:
        data = scale(data, with_mean=preproc_params['with_mean'], with_std=preproc_params['with_std'])
    if preproc_params['threshold_low_var']:
        selector = VarianceThreshold(threshold=preproc_params['threshold_low_var'])
        data = selector.fit_transform(data)
    end = time.time()
    print(f"Prepared data in {(end - start) / 60:.1f} minutes; data of shape {data.shape}")
    return data


def prepare_train_and_test_set(folder_list, preproc_param, test_size=0.2):
    data, labels = prepare_data_and_labels(folder_list, preproc_param)
    data = preprocess_data(data, preproc_param)
    return train_test_split(data, labels, test_size=test_size, random_state=42)


def get_paths_of_image_folders(path_folder):
    images_paths = []
    for root, dirs, files in os.walk(path_folder):
        for name in dirs:
            images_paths.append(path_folder + name)
    return images_paths


def get_folder_name(path_folder):
    name_start = path_folder.rfind('/')
    folder_name = path_folder[name_start + 1:]
    return folder_name


def export_data(data, path):
    np.save(path, data, allow_pickle=False)
    print(f'Saved numpy array to {path}')
    return None


def read_data(path):
    np.load(path, )

import numpy as np
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, IncrementalPCA
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic
import time


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


def preprocess_data(images_list, data_params):
    with_image = data_params['with_image']
    with_binary_patterns = data_params['with_binary_patterns']
    histogram_params = data_params['histogram_params']
    nb_segments = data_params['nb_segments']
    threshold_low_var = data_params['threshold_low_var']
    nb_components_pca = data_params['nb_components_pca']
    batch_size_pca = data_params['batch_size_pca']
    with_mean = data_params['with_mean']
    with_std = data_params['with_std']

    if not (with_image or with_binary_patterns or histogram_params or nb_segments):
        raise ValueError(
            "At least one of 'with_image', 'with_binary_patterns', 'histogram_params', 'nb_segments' has to be True.")
    data = feature_computation(images_list, with_image, with_binary_patterns, histogram_params, nb_segments)
    if threshold_low_var or nb_components_pca or batch_size_pca:
        data = dimension_reduction(data, threshold_low_var, nb_components_pca, batch_size_pca)
    if with_mean or with_std:
        data = scale_data(data, with_mean, with_std)
    return data


def hists_to_arr(histograms_list):
    hist_0 = []
    hist_1 = []
    hist_2 = []
    hist_3 = []
    for hists in histograms_list:
        hist_0.append(hists[0])
        hist_1.append(hists[1])
        hist_2.append(hists[2])
        hist_3.append(hists[3])
    return [np.array(hist_0), np.array(hist_1), np.array(hist_2), np.array(hist_3)]

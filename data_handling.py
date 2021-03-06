import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import IncrementalPCA
from sklearn.utils import shuffle
from skimage.feature import local_binary_pattern
from skimage.transform import rescale
from skimage.segmentation import slic
import time
from typing import Union


def compute_local_binary_pattern(image: np.ndarray, nb_pts=None, radius=3) -> np.ndarray:
    if image.ndim == 2:
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))
    if nb_pts is None:
        nb_pts = 8 * radius
    image_lbp = np.zeros(image.shape)
    for chnl in range(0, image.shape[-1]):
        image_lbp[:, :, chnl] = local_binary_pattern(image[:, :, chnl], nb_pts, radius)
    image_lbp = image_lbp / np.amax(image_lbp)
    return image_lbp


def compute_histograms_from_image(image: np.ndarray, nb_divisions: int, nb_bins: int) -> np.ndarray:
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
                histograms[ch, i * nb_divisions + j, :] = np.histogram(sub_img, bins=nb_bins, density=True)[0]
        histograms[ch, :, :] = histograms[ch, :, :] / np.sum(histograms[ch, :, :])
    return histograms


def segment_image(img: np.ndarray, nb_segments: int) -> np.ndarray:
    return slic(img, n_segments=nb_segments)


def scale_data(data: np.ndarray, with_mean: bool, with_std: bool) -> np.ndarray:
    return scale(data, axis=-1, with_mean=with_mean, with_std=with_std)


def compute_features(images_list: list, with_image: bool, with_binary_patterns: bool, histogram_params: dict,
                     nb_segments: int) -> np.ndarray:
    if not (with_image or with_binary_patterns or histogram_params or nb_segments):
        raise ValueError(
            "At least one of 'with_image', 'with_binary_patterns', 'histogram_params', 'nb_segments' has to be True.")
    start = time.time()
    data = []

    for img in images_list:
        data_img = np.empty(0)
        if with_image:
            img = rescale(img, with_image)
            data_img = np.append(data_img, img.flatten())
        if with_binary_patterns:
            data_img = np.append(data_img, compute_local_binary_pattern(img).flatten())
        if histogram_params:
            nb_divisions, nb_bins = histogram_params
            data_img = np.append(data_img,
                                 compute_histograms_from_image(img, nb_divisions=nb_divisions,
                                                               nb_bins=nb_bins).flatten())
        if nb_segments:
            data_img = np.append(data_img, segment_image(img, nb_segments).flatten())
        data.append(data_img)
    data = np.array(data)
    end = time.time()
    print('Computed features in {:.0f}min.'.format((end - start) // 60))
    return data


def reduce_dimension(data: np.ndarray, nb_components_pca: int, batch_size_pca: int) -> np.ndarray:
    start = time.time()
    old_shape = data.shape
    if nb_components_pca:
        pca = IncrementalPCA(n_components=nb_components_pca, batch_size=batch_size_pca)
        data = pca.fit_transform(data)
        end = time.time()
        print('Dimensionality reduction took {:.0f}min; reduction from {} to {}'.format((end - start) // 60,
                                                                                             old_shape, data.shape))
    return data


def remove_low_var_features(data: np.ndarray, threshold_low_var: float) -> np.ndarray:
    start_time = time.time()
    if threshold_low_var:
        selector = VarianceThreshold(threshold=threshold_low_var)
        data = selector.fit_transform(data)
        end_time = time.time()
        print('Removed low var features in {:.0f}min.'.format((end_time - start_time) // 60))
    return data


def preprocess_images(images_list: list, data_params: dict) -> np.ndarray:
    with_image = data_params['with_image']
    with_binary_patterns = data_params['with_binary_patterns']
    histogram_params = data_params['histogram_params']
    nb_segments = data_params['nb_segments']
    threshold_low_var = data_params['threshold_low_var']
    nb_components_pca = data_params['nb_components_pca']
    batch_size_pca = data_params['batch_size_pca']

    data = compute_features(images_list, with_image, with_binary_patterns, histogram_params, nb_segments)
    data = remove_low_var_features(data, threshold_low_var)
    data = reduce_dimension(data, nb_components_pca, batch_size_pca)

    return data


def rearrange_hists(histograms_list: list, data_params: dict, read_hist: bool) -> np.ndarray:
    start_time = time.time()
    hist_hsl = data_params['hist_hsl']
    hist_h = data_params['hist_h']
    hist_s = data_params['hist_s']
    hist_l = data_params['hist_l']

    data = []
    for hists in histograms_list:
        data_hist = np.empty(0)
        if hist_hsl:
            data_hist = np.append(data_hist, hists[0].flatten())
        if hist_h:
            data_hist = np.append(data_hist, hists[1])
        if hist_s:
            data_hist = np.append(data_hist, hists[2])
        if hist_l:
            data_hist = np.append(data_hist, hists[3])
        if read_hist == 'context':
            if hist_hsl:
                data_hist = np.append(data_hist, hists[4].flatten())
            if hist_h:
                data_hist = np.append(data_hist, hists[5])
            if hist_s:
                data_hist = np.append(data_hist, hists[6])
            if hist_l:
                data_hist = np.append(data_hist, hists[7])
        data.append(data_hist)
    data = np.array(data)
    end_time = time.time()
    print('Rearranged histograms in {}s; histograms of shape {}'.format((end_time - start_time), data.shape))
    return data


def split_and_sample_data(data: np.ndarray, labels: np.ndarray, paths_imgs: list, test_size: Union[None, float]) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list):
    seed = 42
    if test_size is not None:
        X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(data,
                                                                                     labels,
                                                                                     paths_imgs,
                                                                                     test_size=test_size,
                                                                                     random_state=seed,
                                                                                     stratify=labels)
    else:
        X_train, y_train, paths_train = shuffle(data, labels, paths_imgs, random_state=seed)
        X_test = np.array([])
        y_test = np.array([])
        paths_test = None
    print('Data before sampling: {} positive, {} total.'.format(np.sum(labels), labels.size))
    print('Training data: {} positive, {} total'.format(np.sum(y_train, dtype=np.uint), y_train.size))
    print('Test data: {} positive, {} total'.format(np.sum(y_test, dtype=np.uint), y_test.size))
    return X_train, X_test, y_train, y_test, paths_train, paths_test


def compute_prior_weight(y_unbalanced: np.ndarray, y_balanced: np.ndarray) -> (float, float):
    p_unbalanced_mite = np.sum(y_unbalanced) / y_unbalanced.size
    p_balanced_mite = np.sum(y_balanced) / y_balanced.size
    factor_mite = p_unbalanced_mite / p_balanced_mite
    factor_no_mite = (1 - p_unbalanced_mite) / (1 - p_balanced_mite)
    return factor_mite, factor_no_mite


def compute_quadratic_features(data: np.ndarray) -> np.ndarray:
    data_squared = np.square(data)
    data = np.append(data, data_squared, axis=1)
    return data

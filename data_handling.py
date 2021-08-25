import numpy as np
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import IncrementalPCA
from skimage.feature import local_binary_pattern
from skimage.transform import rescale
from skimage.segmentation import slic
import random
import time


def compute_local_binary_pattern(image, nb_pts=None, radius=3):
    if image.ndim == 2:
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))
    if nb_pts is None:
        nb_pts = 8 * radius
    image_lbp = np.zeros(image.shape)
    for ch in range(0, image.shape[-1]):
        image_lbp[:, :, ch] = local_binary_pattern(image[:, :, ch], nb_pts, radius)
    image_lbp = image_lbp / np.amax(image_lbp)
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
                histograms[ch, i * nb_divisions + j, :] = np.histogram(sub_img, bins=nb_bins, density=True)[0]
        histograms[ch, :, :] = histograms[ch, :, :] / np.sum(histograms[ch, :, :])
    return histograms


def segment_image(img, nb_segments):
    return slic(img, n_segments=nb_segments)


def scale_data(data, with_mean, with_std):
    return scale(data, axis=-1, with_mean=with_mean, with_std=with_std)


def feature_computation(images_list, with_image, with_binary_patterns, histogram_params, nb_segments):
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
                                 compute_histograms(img, nb_divisions=nb_divisions, nb_bins=nb_bins).flatten())
        if nb_segments:
            data_img = np.append(data_img, segment_image(img, nb_segments).flatten())
        data.append(data_img)
    data = np.array(data)
    end = time.time()
    print('Computed features in {:.1f} minutes.'.format((end - start) / 60))
    return data


def dimension_reduction(data, nb_components_pca, batch_size_pca):
    start = time.time()
    old_shape = data.shape
    if nb_components_pca:
        pca = IncrementalPCA(n_components=nb_components_pca, batch_size=batch_size_pca)
        # data = normalize(data)
        data = pca.fit_transform(data)
        end = time.time()
        print(
            'Dimensionality reduction took {:.1f} minutes; reduction from {} to {}'.format((end - start) / 60,
                                                                                           old_shape, data.shape))
    return data


def remove_low_var_features(data, threshold_low_var):
    start_time = time.time()
    if threshold_low_var:
        selector = VarianceThreshold(threshold=threshold_low_var)
        data = selector.fit_transform(data)
        end_time = time.time()
        print('Removed low var features in {:.1f} minutes.'.format((end_time - start_time) / 60))
    return data


def preprocess_images(images_list, data_params):
    with_image = data_params['with_image']
    with_binary_patterns = data_params['with_binary_patterns']
    histogram_params = data_params['histogram_params']
    nb_segments = data_params['nb_segments']
    threshold_low_var = data_params['threshold_low_var']
    nb_components_pca = data_params['nb_components_pca']
    batch_size_pca = data_params['batch_size_pca']

    data = feature_computation(images_list, with_image, with_binary_patterns, histogram_params, nb_segments)
    data = remove_low_var_features(data, threshold_low_var)
    data = dimension_reduction(data, nb_components_pca, batch_size_pca)

    return data


def rearrange_hists(histograms_list, data_params, read_hist):
    start_time = time.time()
    hist_hsl = data_params['hist_hsl']
    hist_h = data_params['hist_h']
    hist_s = data_params['hist_s']
    hist_l = data_params['hist_l']

    data = []
    while histograms_list != []:
        data_hist = np.empty(0)
        hists = histograms_list.pop(0)
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
        del hists
    data = np.array(data)
    end_time = time.time()
    print('Rearranged histograms in {:.1f}s; histograms of shape {}'.format((end_time - start_time), data.shape))
    return data


def downsize_false_candidates(data, labels, paths_images, percentage_true):
    nb_candidates = labels.size
    nb_true_cand = np.sum(labels)
    if percentage_true is None:
        return data, labels, paths_images
    elif nb_true_cand / nb_candidates <= percentage_true:
        nb_false_remove = nb_candidates - int(nb_true_cand / percentage_true)
        idxs_false = list(np.arange(0, nb_candidates)[labels == 0])
        random.seed(42)  # to assure, same sample is drawn; remove if selection should be random
        idxs_false_remove = random.sample(idxs_false, k=nb_false_remove)
        print('Before downsizing: {} candidates; {} mites.'.format(labels.size, np.sum(labels)))
        data = np.delete(data, idxs_false_remove, axis=0)
        labels = np.delete(labels, idxs_false_remove)
        paths_images = np.delete(paths_images, idxs_false_remove)
        print('After downsizing: {} candidates; {} mites.'.format(labels.size, np.sum(labels)))
        return data, labels, paths_images
    else:
        raise ValueError(
            'Ratio of true candidates {} to total candidates {} is already greater than {}'.format(nb_true_cand,
                                                                                                   nb_candidates,
                                                                                                   percentage_true))

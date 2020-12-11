import os
from pathlib import Path
import numpy as np
import re
from skimage.io import imread, imshow
from sklearn.preprocessing import scale
from skimage.color import rgb2ycbcr, ycbcr2rgb
from skimage.feature import local_binary_pattern, haar_like_feature, haar_like_feature_coord, draw_haar_like_feature
from skimage.transform import integral_image
from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity


def read_data_from_folder(path_image_folder, gray_scale=False, normalize_hist=True, binary_patterns=False, haar_features=False):
    image_folder = Path(path_image_folder).rglob('*.jpg')
    images_paths = [str(path) for path in image_folder]
    nb_images = len(images_paths)
    data = []
    labels = []

    pattern_true = r'true'
    pattern_false = r'false'
    for i in range(0, nb_images):
        image = imread(images_paths[i], as_gray=gray_scale)
        if normalize_hist:
            image = histogram_equalization(image)
        if binary_patterns:
            compute_local_binary_pattern()
        if haar_features:
            compute_haar_features()

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
            'Image has neither one channel (gray scale) nor three channels (RGB); got shape {} instead.'.format(
                image.shape))


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
    folder_name = path_folder[name_start + 1:]
    return folder_name


def compute_local_binary_pattern(image, nb_pts=None, radius=3):
    if image.ndim != 2:
        raise ValueError('Image must be 2-dimensional, got {}-dimensional image'.format(image.ndim))
    if nb_pts is None:
        nb_pts = 8 * radius
    image_lbp = local_binary_pattern(image, nb_pts, radius)
    return image_lbp


def compute_haar_features(image, feature_type=None, feature_coord=None):
    if image.ndim != 2:
        raise ValueError('Image must be 2-dimensional, got {}-dimensional image'.format(image.ndim))
    int_img = integral_image(image)
    feature_vector = haar_like_feature(int_img, 0, 0, image.shape[0], image.shape[1], feature_type=feature_type,
                                       feature_coord=feature_coord)
    return feature_vector

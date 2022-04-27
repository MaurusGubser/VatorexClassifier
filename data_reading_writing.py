import os
from pathlib import Path
import re
from typing import Union
import numpy as np
from skimage.io import imread
from skimage.util import img_as_ubyte
from skimage.exposure import equalize_adapthist
import time

from data_handling import preprocess_images, rearrange_hists, scale_data, compute_quadratic_features


def read_histograms(path: str) -> list:
    with open(path, 'r') as read_file:
        for line in read_file.readlines():
            line_tokens = line.replace('\n', '').split(':')
            if line_tokens[0] == 'size(h,l,s)':
                size_h = int(line_tokens[1])
                size_l = int(line_tokens[2])
                size_s = int(line_tokens[3])
                pdf_hls = np.zeros((size_h, size_l, size_s), dtype='float64')
                pdf_h = np.zeros(size_h, dtype='float64')
                pdf_l = np.zeros(size_l, dtype='float64')
                pdf_s = np.zeros(size_s, dtype='float64')
                mode = 'hls'
            elif line_tokens[0] in ['h', 'l', 's']:
                mode = line_tokens[0]
            elif mode == 'hls':
                pdf_hls[int(line_tokens[0]), int(line_tokens[1]), int(line_tokens[2])] = np.float64(line_tokens[3])
            elif mode == 'h':
                pdf_h[int(line_tokens[0])] = np.float64(line_tokens[1])
            elif mode == 'l':
                pdf_l[int(line_tokens[0])] = np.float64(line_tokens[1])
            elif mode == 's':
                pdf_s[int(line_tokens[0])] = np.float64(line_tokens[1])
            else:
                raise ValueError('Line {} does not correspond to expected pattern.'.format(line))
    histograms = [pdf_hls, pdf_h, pdf_l, pdf_s]
    return histograms


def read_data_from_single_dir(path_folder: str, read_image: bool, read_hist: Union[None, str],
                              with_false1: bool) -> (list, list, list, list):
    if not read_image and read_hist not in ['candidate', 'context']:
        raise AssertionError('Got invalid values for read_image and read_hist; {} and {}'.format(read_image, read_hist))
    images_paths = [str(path) for path in Path(path_folder).glob('*.jpg')]
    histograms_paths = [path.replace('.jpg', '.hist') for path in images_paths]
    histograms_context_paths = [path.replace('.jpg', '_context.hist') for path in images_paths]
    images = []
    histograms = []
    labels = []
    pattern_true = r'true'
    pattern_false1 = r'false1'
    pattern_false2 = r'false2'

    for path_img, path_hist, path_hist_context in zip(images_paths, histograms_paths, histograms_context_paths):
        if read_image:
            image = img_as_ubyte(equalize_adapthist(imread(path_img, as_gray=False)))
            images.append(image)
        if read_hist == 'candidate':
            histograms.append(read_histograms(path_hist))
        if read_hist == 'context':
            histograms.append(read_histograms(path_hist) + read_histograms(path_hist_context))
        if re.search(pattern_true, path_img):
            labels.append(1)
        elif re.search(pattern_false1, path_img):
            labels.append(2)
        elif re.search(pattern_false2, path_img):
            labels.append(0)
        else:
            raise AssertionError('Image path {} does not contain true, false1 or false2.'.format(path_img))
    if with_false1:
        for i in range(0, len(labels)):
            if labels[i] == 2:
                labels[i] = 0
    else:
        mask = [i != 2 for i in labels]
        if read_hist:
            histograms = [histograms[i] for i in range(0, len(labels)) if mask[i]]
        if read_image:
            images = [images[i] for i in range(0, len(labels)) if mask[i]]
        images_paths = [images_paths[i] for i in range(0, len(labels)) if mask[i]]
        labels = [label for label in labels if label != 2]
    return images, histograms, labels, images_paths


def get_folder_name(path_folder: str) -> str:
    name_start = path_folder[:len(path_folder) - 1].rfind('/')
    folder_name = path_folder[name_start + 1:]
    folder_name = folder_name.replace('/', '')
    return folder_name


def set_export_data_name(folder_name: str, data_params: dict) -> str:
    name = folder_name
    for value in data_params.values():
        name = name + '_' + str(value)
    return name


def export_data(data_img: np.ndarray, data_hist: np.ndarray, labels: np.ndarray, paths_images: np.ndarray,
                data_name: str) -> None:
    path_export = 'Preprocessed_Data/' + data_name
    if os.path.exists(path_export + '.npz'):
        print('Preprocessed data with these parameters already exported.')
        return None
    if not os.path.exists('Preprocessed_Data'):
        os.mkdir('Preprocessed_Data')
    np.savez(path_export,
             img=data_img,
             hist=data_hist,
             labels=labels,
             paths_images=paths_images)
    print('Saved data and labels in files {}'.format(path_export))
    return None


def reload_data_and_labels(path_data: str) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    data = np.load(path_data)
    images = data['img']
    histograms = data['hist']
    labels = data['labels']
    paths_images = data['paths_images']
    return images, histograms, labels, paths_images


def read_data_and_labels_from_path(path: str, data_params: dict) -> (np.ndarray, np.ndarray, np.ndarray):
    folder_name = get_folder_name(path)
    path_preprocessed = 'Preprocessed_Data/' + set_export_data_name(folder_name, data_params) + '.npz'
    read_image = data_params['read_image']
    read_hist = data_params['read_hist']
    with_false1 = data_params['with_false1']
    if os.path.exists(path_preprocessed):
        data_images, data_histograms, labels, paths_images = reload_data_and_labels(path_preprocessed)
        print('Re-loaded preprocessed data and labels from {}'.format(path_preprocessed))
    else:
        folder_list = [str(p) for p in list(Path(path).rglob('extracted'))]
        data_images, data_histograms, labels, paths_images = [], [], [], []
        for path_series in folder_list:
            imgs, hists, lbls, imgs_paths = read_data_from_single_dir(path_series, read_image, read_hist, with_false1)
            data_images += imgs
            data_histograms += hists
            labels += lbls
            imgs_paths += imgs_paths
    if read_image:
        data_images = preprocess_images(data_images, data_params)
    if read_hist in ['candidate', 'context']:
        data_histograms = rearrange_hists(data_histograms, data_params, read_hist)
    labels = np.array(labels)
    paths_images = np.array(paths_images)
    data_name = set_export_data_name(folder_name, data_params)
    export_data(data_images, data_histograms, labels, paths_images, data_name)
    data = concatenate_data(data_images, data_histograms, read_image, read_hist)
    print('Data before preprocessing of shape {}'.format(data.shape))
    if data_params['quadratic_features']:
        data = compute_quadratic_features(data)
    if data_params['with_mean'] or data_params['with_std']:
        data = scale_data(data, data_params['with_mean'], data_params['with_mean'])
    print('Data after preprocessing of shape {}'.format(data.shape))
    return data, labels, paths_images


def concatenate_data(data_img: np.ndarray, data_hist: np.ndarray, read_image: bool, read_hist: bool) -> np.ndarray:
    if not read_image:
        data = data_hist
    elif not read_hist:
        data = data_img
    else:
        data = np.append(data_img, data_hist, axis=1)
    return data

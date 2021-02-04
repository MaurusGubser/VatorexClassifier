import os
from pathlib import Path
import re
import numpy as np
from skimage.io import imread
from skimage.exposure import equalize_adapthist

from data_handling import preprocess_data


def hist_read(path):
    with open(path, 'r') as read_file:
        for line in read_file.readlines():
            line_tokens = line.replace('\n', '').split(':')
            if line_tokens[0] == 'size(h,l,s)':
                size_h = int(line_tokens[1])
                size_l = int(line_tokens[2])
                size_s = int(line_tokens[3])
                pdf_hls = np.zeros((size_h, size_l, size_s))
                pdf_h = np.zeros(size_h)
                pdf_l = np.zeros(size_l)
                pdf_s = np.zeros(size_s)
                mode = 'hls'
            elif line_tokens[0] in ['h', 'l', 's']:
                mode = line_tokens[0]
            elif mode == 'hls':
                pdf_hls[int(line_tokens[0]), int(line_tokens[1]), int(line_tokens[2])] = float(line_tokens[3])
            elif mode == 'h':
                pdf_h[int(line_tokens[0])] = float(line_tokens[1])
            elif mode == 'l':
                pdf_l[int(line_tokens[0])] = float(line_tokens[1])
            elif mode == 's':
                pdf_s[int(line_tokens[0])] = float(line_tokens[1])
            else:
                raise ValueError(f'Line {line} does not correspond to expected pattern.')
    histograms = [pdf_hls, pdf_h, pdf_l, pdf_s]
    return histograms


def read_images_hist_from_folder(path_folder):
    images_folder = Path(path_folder).rglob('*.jpg')
    histograms_folder = Path(path_folder).rglob('*.hist')
    images_paths = [str(path) for path in images_folder]
    histogram_paths = [str(path) for path in histograms_folder]
    images = []
    histograms = []
    labels = []
    pattern_true = r'true'
    pattern_false = r'false'
    for path_img, path_hist in zip(images_paths, histogram_paths):
        image = imread(path_img, as_gray=False)
        image = equalize_adapthist(image)
        images.append(image)
        histograms.append(hist_read(path_hist))
        if re.search(pattern_true, path_img) and re.search(pattern_true, path_hist):
            labels.append(1)
        elif re.search(pattern_false, path_img) and re.search(pattern_false, path_hist):
            labels.append(0)
        else:
            raise AssertionError(f"Image path {path_img} and histogram path {path_hist} are not compatible.")
    return images, histograms, labels


def read_images_and_histograms(folder_list):
    images = []
    histograms = []
    labels = []
    for folder_path in folder_list:
        imgs, hists, lbls = read_images_hist_from_folder(folder_path)
        images = images + imgs
        histograms = histograms + hists
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
        images_list, histograms_list, labels_list = read_images_and_histograms(folder_list)
        labels = np.array(labels_list)
        data = preprocess_data(images_list, preprocessing_params)
        data_name = set_export_data_name(folder_name, preprocessing_params)
        export_data(data, labels, data_name)
        return data, labels

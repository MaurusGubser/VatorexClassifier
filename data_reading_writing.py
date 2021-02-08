import os
from pathlib import Path
import re
import numpy as np
from skimage.io import imread
from skimage.util import img_as_ubyte
from skimage.exposure import equalize_adapthist
import time

from data_handling import preprocess_images, rearrange_hists, scale_data


def hist_read(path):
    with open(path, 'r') as read_file:
        for line in read_file.readlines():
            line_tokens = line.replace('\n', '').split(':')
            if line_tokens[0] == 'size(h,l,s)':
                size_h = int(line_tokens[1])
                size_l = int(line_tokens[2])
                size_s = int(line_tokens[3])
                pdf_hls = np.zeros((size_h, size_l, size_s), dtype='float32')
                pdf_h = np.zeros(size_h, dtype='float32')
                pdf_l = np.zeros(size_l, dtype='float32')
                pdf_s = np.zeros(size_s, dtype='float32')
                mode = 'hls'
            elif line_tokens[0] in ['h', 'l', 's']:
                mode = line_tokens[0]
            elif mode == 'hls':
                pdf_hls[int(line_tokens[0]), int(line_tokens[1]), int(line_tokens[2])] = np.float32(line_tokens[3])
            elif mode == 'h':
                pdf_h[int(line_tokens[0])] = np.float32(line_tokens[1])
            elif mode == 'l':
                pdf_l[int(line_tokens[0])] = np.float32(line_tokens[1])
            elif mode == 's':
                pdf_s[int(line_tokens[0])] = np.float32(line_tokens[1])
            else:
                raise ValueError(f'Line {line} does not correspond to expected pattern.')
    histograms = [pdf_hls, pdf_h, pdf_l, pdf_s]
    return histograms


def read_images_hist_from_folder(path_folder, read_image, read_hist):
    histograms_paths = [str(path) for path in Path(path_folder).rglob('*.hist')]
    images_paths = [path.replace('.hist', '.jpg') for path in histograms_paths]

    images = []
    histograms = []
    labels = []
    pattern_true = r'true'
    pattern_false = r'false'

    for path_img, path_hist in zip(images_paths, histograms_paths):
        if read_image:
            image = img_as_ubyte(equalize_adapthist(imread(path_img, as_gray=False)))
            images.append(image)
        if read_hist:
            histograms.append(hist_read(path_hist))
        if re.search(pattern_true, path_img) and re.search(pattern_true, path_hist):
            labels.append(1)
        elif re.search(pattern_false, path_img) and re.search(pattern_false, path_hist):
            labels.append(0)
        else:
            raise AssertionError(f'Label of histogram path {path_hist} and image path {path_img} are not compatible')

    return images, histograms, labels


def read_images_and_histograms(folder_list, read_image, read_hist):
    start_time = time.time()
    images = []
    histograms = []
    labels = []
    for folder_path in folder_list:
        imgs, hists, lbls = read_images_hist_from_folder(folder_path, read_image, read_hist)
        images = images + imgs
        histograms = histograms + hists
        labels = labels + lbls
    end_time = time.time()
    print(f'Read images and histograms in {(end_time - start_time)/60:.1f} minutes')
    return images, histograms, labels


def get_paths_of_image_folders(path_folder):
    folder_list = []
    for root, dirs, files in os.walk(path_folder):
        if root != path_folder:
            continue
        for name in dirs:
            folder_list.append(path_folder + name)
    return folder_list


def get_folder_name(path_folder):
    name_start = path_folder[:len(path_folder) - 1].rfind('/')
    folder_name = path_folder[name_start + 1:]
    folder_name = folder_name.replace('/', '')
    return folder_name


def set_export_data_name(folder_name, data_params):
    name = folder_name
    for value in data_params.values():
        name = name + '_' + str(value)
    return name


def export_data(data_img, data_hist_list, labels, data_name):
    path_export = 'Preprocessed_Data/' + data_name
    if os.path.exists(path_export + '.npz'):
        print('Preprocessed data with these parameters already exported.')
        return None
    if not os.path.exists('Preprocessed_Data'):
        os.mkdir('Preprocessed_Data')
    np.savez(path_export,
             img=data_img,
             hist_0=data_hist_list[0],
             hist_1=data_hist_list[1],
             hist_2=data_hist_list[2],
             hist_3=data_hist_list[3],
             labels=labels)
    print(f'Saved data and labels in files {path_export}')
    return None


def load_data_and_labels(path_data):
    data = np.load(path_data)
    images = data['img']
    hist_0 = data['hist_0']
    hist_1 = data['hist_1']
    hist_2 = data['hist_2']
    hist_3 = data['hist_3']
    labels = data['labels']
    return images, [hist_0, hist_1, hist_2, hist_3], labels


def read_data_and_labels(path, data_params):
    folder_name = get_folder_name(path)
    path_preprocessed = 'Preprocessed_Data/' + set_export_data_name(folder_name, data_params) + '.npz'
    read_image = data_params['read_image']
    read_hist = data_params['read_hist']
    if os.path.exists(path_preprocessed):
        data_images, data_histograms, labels = load_data_and_labels(path_preprocessed)
        data = concatenate_data(data_images, data_histograms, read_image, read_hist)
        del data_images
        del data_histograms
        print(f'Re-loaded preprocessed data and labels from {path_preprocessed}')
        if data_params['with_mean'] or data_params['with_std']:
            data = scale_data(data, data_params['with_mean'], data_params['with_mean'])
        return data, labels
    else:
        folder_list = get_paths_of_image_folders(path)
        data_images, data_histograms, labels = read_images_and_histograms(folder_list, read_image, read_hist)
        if read_image:
            data_images = preprocess_images(data_images, data_params)
        if read_hist:
            data_histograms = rearrange_hists(data_histograms)
        labels = np.array(labels)

        data_name = set_export_data_name(folder_name, data_params)
        export_data(data_images, data_histograms, labels, data_name)
        data = concatenate_data(data_images, data_histograms, read_image, read_hist)
        del data_images
        del data_histograms
        if data_params['with_mean'] or data_params['with_std']:
            data = scale_data(data, data_params['with_mean'], data_params['with_mean'])
        return data, labels


def concatenate_data(data_img, data_hist, read_image, read_hist):
    if not read_image:
        data = np.concatenate((data_hist[0], data_hist[1], data_hist[2], data_hist[3]), axis=1)
    elif not read_hist:
        data = data_img
    else:
        data_hist = np.concatenate(
            (data_hist[0].flatten(), data_hist[1], data_hist[2], data_hist[3]), axis=1)
        data = np.append(data_img, data_hist, axis=1)
    return data

import matplotlib.pyplot as plt
import numpy as np
from data_reading_writing import read_images_hist_from_folder


def get_last_path_str(path: str) -> str:
    start = path.rfind('/') + 1
    name = path[start:]
    return name


def plot_images_and_hist(path_folder: str) -> None:
    images, hist, labels, images_paths = read_images_hist_from_folder(path_folder, True, 'candidate')
    labels = np.array(labels)
    nb_true = np.sum(labels)
    nb_false = labels.size - nb_true
    nb_figs = min(nb_false, nb_true, 5)
    idx_true = np.where(labels == 1)[0]
    idx_false = np.where(labels == 0)[0]

    hist_labels = ['H', 'L', 'S']
    for i in range(0, nb_figs):
        fig, axs = plt.subplots(nrows=2, ncols=4)
        axs[0, 0].imshow(images[idx_true[i]])
        axs[0, 0].set_title(get_last_path_str(images_paths[idx_true[i]]))
        axs[1, 0].imshow(images[idx_false[i]])
        axs[1, 0].set_title(get_last_path_str(images_paths[idx_false[i]]))
        for j in range(1, 4):
            axs[0, j].plot(hist[idx_true[i]][j], label=hist_labels[j - 1])
            axs[0, j].grid(axis='both')
            axs[0, j].legend()
            axs[1, j].plot(hist[idx_false[i]][j], label=hist_labels[j - 1])
            axs[1, j].grid(axis='both')
            axs[1, j].legend()
        fig.suptitle(get_last_path_str(path_folder))
        fig.tight_layout()
    plt.show()
    return None


path_folder = 'Candidate_Images/Mite4_relabelledtol05/200328-R02(43Milben,labeled)/'
plot_images_and_hist(path_folder)

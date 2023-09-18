import torch
import random

import numpy as np

from skimage import measure


def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def centroid_from_masks(M):

    C = []
    for i in np.arange(0, M.shape[0]):
        m = np.squeeze(M[i, :, :])
        labels = measure.label(m)
        props = measure.regionprops(labels)[0]
        C.append([props.centroid])

    return np.squeeze(np.array(C))


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def distance_mitosis(ref, pred):

    d = np.sqrt(np.sum(np.square(ref-pred)))

    return d

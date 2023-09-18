import torchvision
import imutils
import numpy as np
import random
import os
import torch
import kornia

from skimage import io

from data.utils import ColourNormalization
from data.constants import *

import matplotlib.pyplot as plt
plt.interactive(False)


class Dataset(object):
    def __init__(self, dataset_id, partition='train', input_shape=(3, 512, 512), labels=1, preallocate=True,
                 stain_normalization=False, save_normalized=True, dir_images='images_norm', dir_masks='dir_masks',
                 only_one_mitosis=False):

        if "TUPAC16" in dataset_id:
            val_id, test_id = TUPAC16_ID_VAL, TUPAC16_ID_TEST
            dir_dataset = PATH_TUPAC_PROCESSED
        elif "MITOS14" in dataset_id:
            val_id, test_id = MITOS14_ID_VAL, MITOS14_ID_TEST
            dir_dataset = PATH_MITOS14_PROCESSED
        elif "MIDOG21" in dataset_id:
            val_id, test_id = MIDOG21_ID_VAL, MIDOG21_ID_TEST
            dir_dataset = PATH_MIDOG21_PROCESSED
        else:
            print("Processing dataset not valid... ")
            return

        self.dir_dataset = dir_dataset
        self.partition = partition
        self.resize = torchvision.transforms.Resize((input_shape[-2], input_shape[-1]))
        self.labels = labels
        self.preallocate = preallocate
        self.input_shape = input_shape
        self.stain_normalization = stain_normalization
        self.save_normalized = save_normalized
        self.dir_images = dir_images
        self.dir_masks = dir_masks
        self.only_one_mitosis = only_one_mitosis

        self.images = os.listdir(dir_dataset + self.dir_images + '/')

        if self.stain_normalization:
            target_stain_patch_path = "../data/color_norm/01.jpg"
            self.color_normalization_function = ColourNormalization(target_stain_patch_path)

            if self.save_normalized:
                if not os.path.isdir(dir_dataset + 'images_norm/'):
                    os.mkdir(dir_dataset + 'images_norm/')

        # Filter wrong files
        self.images = [ID for ID in self.images if ID != 'Thumbs.db']
        self.images = self.images

        if self.partition == 'train':
            idx = np.in1d([ID.split('_')[0] for ID in self.images], val_id + test_id)
            self.images = [self.images[i] for i in range(self.images.__len__()) if not idx[i]]
        elif self.partition == 'val':
            idx = np.in1d([ID.split('_')[0] for ID in self.images], val_id)
            self.images = [self.images[i] for i in range(self.images.__len__()) if idx[i]]
        elif self.partition == 'test':
            idx = np.in1d([ID.split('_')[0] for ID in self.images], test_id)
            self.images = [self.images[i] for i in range(self.images.__len__()) if idx[i]]
        else:
            print('Wrong partition', end='\n')

        if self.preallocate:
            # Pre-load images
            self.X = np.zeros((len(self.images), input_shape[0], input_shape[1], input_shape[2]))
            self.M = np.zeros((len(self.images), input_shape[1], input_shape[2]))
            self.Y = np.zeros((len(self.images), labels))
            self.N = np.zeros((len(self.images), 1))

            for iImage in np.arange(0, len(self.images)):
                print(str(iImage + 1) + '/' + str(len(self.images)), end='\r')

                im = np.array(io.imread(dir_dataset + self.dir_images + '/' + self.images[iImage]))
                im = imutils.resize(im, height=self.input_shape[1])

                # Stain Normalization
                if self.stain_normalization:
                    if not os.path.exists(dir_dataset + 'images_norm/' + self.images[iImage]):
                        im = self.color_normalization_function(im)
                        if self.save_normalized:
                            io.imsave(dir_dataset + 'images_norm/' + self.images[iImage], np.uint8(im))

                im = np.transpose(im, (2, 0, 1))
                # Intensity normalization
                im = im / 255

                if os.path.isfile(dir_dataset + self.dir_masks + '/' + self.images[iImage]):
                    mask = np.array(io.imread(dir_dataset + self.dir_masks + '/' + self.images[iImage]))
                    mask = imutils.resize(mask, height=self.input_shape[1]) / 255
                    mask = np.double(mask > 0)
                else:
                    mask = np.zeros((self.input_shape[1], self.input_shape[1]))

                self.X[iImage, :, :, :] = im
                self.M[iImage, :, :] = mask
                self.Y[iImage, :] = np.max(mask[:, :])
                self.N[iImage, :] = np.sum(mask[:, :])

        # Quit training samples with more than one mitosis
        if self.partition == 'train' and self.only_one_mitosis:
            idx = np.squeeze(np.argwhere(np.squeeze(self.N) <= 1))
            self.filter_cases(idx)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

    def __getitem__(self, index):
        'Generates one sample of data'

        x = self.X[index, :, :, :].astype(np.float32)
        m = self.M[index, :, :].astype(np.int)
        y = self.Y[index, :].astype(np.int)

        return x, y, m

    def filter_cases(self, indexes):
        self.X = self.X[indexes, :, :, :]
        self.M = self.M[indexes, :, :]
        self.Y = self.Y[indexes, :]
        self.N = self.N[indexes, :]
        self.images = [self.images[i] for i in range(self.images.__len__()) if i in indexes]


class Generator(object):
    def __init__(self, dataset, batch_size, shuffle=True, balance=False, strong_augmentation=True):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(0, len(self.dataset.images))
        self._idx = 0
        self.balance = balance
        self.strong_augmentation = strong_augmentation
        self.augmentations = AugmentationsSegmentation(strong_augmentation=strong_augmentation)

        if self.balance:
            self.indexes = balance_dataset(self.indexes, self.dataset.Y.flatten())
        self._reset()

    def __len__(self):
        N = self.dataset.X.shape[0]
        b = self.batch_size
        return N // b

    def __iter__(self):
        return self

    def __next__(self):

        if self._idx + self.batch_size > self.dataset.X.shape[0]:
            self._reset()
            raise StopIteration()

        # Load images and include into the batch
        X, Y, M = [], [], []
        for i in range(self._idx, self._idx + self.batch_size):
            x, y, m = self.dataset.__getitem__(self.indexes[i])

            X.append(np.expand_dims(x, axis=0))
            Y.append(np.expand_dims(y, axis=0))
            M.append(np.expand_dims(m, axis=0))

        # Update index iterator
        self._idx += self.batch_size

        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        M = np.concatenate(M, axis=0)

        return X, Y, M

    def _reset(self):
        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0


def balance_dataset(indexes, Y):
    classes = [0, 1]
    counts = np.bincount(Y.astype(np.int))

    upsampling = [round(np.max(counts)/counts[iClass]) for iClass in classes]

    indexes_new = []
    for iClass in classes:
        if upsampling[iClass] == 1:
            indexes_iclass = indexes[Y == classes[iClass]]
        else:
            indexes_iclass = np.random.choice(indexes[Y == classes[iClass]], counts[iClass]*upsampling[iClass])
        indexes_new.extend(indexes_iclass)

    indexes_new = np.array(indexes_new)

    return indexes_new


class AugmentationsSegmentation(torch.nn.Module):
    def __init__(self, strong_augmentation=False):
        super(AugmentationsSegmentation, self).__init__()

        self.strong_augmentation = strong_augmentation

        # we define and cache our operators as class members
        self.kHor = kornia.augmentation.RandomHorizontalFlip(p=0.5)
        self.kVert = kornia.augmentation.RandomVerticalFlip(p=0.5)
        self.kAffine = kornia.augmentation.RandomRotation(p=0.5, degrees=[-90, 90])
        self.kTransp = RandomTranspose(p=0.5)

        if self.strong_augmentation:
            self.kElastic = kornia.augmentation.RandomElasticTransform(p=0.5)

    def forward(self, img, mask):
        img_out = img

        # Apply geometric tranform
        img_out = self.kTransp(self.kAffine(self.kVert(self.kHor(img_out))))

        # Infer geometry params to mask
        mask_out = self.kTransp(self.kAffine(self.kVert(self.kHor(mask, self.kHor._params), self.kVert._params), self.kAffine._params), self.kTransp._params)

        if self.strong_augmentation:
            img_out = self.kElastic(img_out)
            mask_out = self.kElastic(mask_out, self.kElastic._params)

        return img_out, mask_out


class RandomTranspose(torch.nn.Module):
    def __init__(self, p):
        super(RandomTranspose, self).__init__()
        self.p = p
        self._params = 0

    def forward(self, x, params=None):
        # Get random state for the operation
        if params is None:
            p = random.random()
            self._params = p
        else:
            p = self._params
        # Apply transform
        if p > 0.5:
            return torch.transpose(x, -2, -1)
        else:
            return x
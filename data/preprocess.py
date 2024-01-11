import json
import os
import argparse

import pandas as pd
import numpy as np

from skimage import io
from utils import ColourNormalization
from constants import *
from PIL import Image


def patch_tupac(patch_size=500):

    folds = os.listdir(PATH_TUPAC_RAW_IMAGES)

    if not os.path.exists(PATH_TUPAC_PROCESSED + 'images/'):
        os.makedirs(PATH_TUPAC_PROCESSED + 'images/')
    if not os.path.exists(PATH_TUPAC_PROCESSED + 'masks/'):
        os.makedirs(PATH_TUPAC_PROCESSED + 'masks/')

    c = 0
    for i_fold in folds:  # e.g. ["mitoses_train_image_data_part_1", "mitoses_train_image_data_part_2"]
        c += 1

        cases, cc = os.listdir(PATH_TUPAC_RAW_IMAGES + i_fold + '/'), 0
        for i_case in cases:
            cc += 1
            print("Batch " + str(c) + "/" + str(len(folds)) + " - Case " + str(cc) + "/" + str(len(cases)) +
                  " - ID: " + str(i_case), end='\n')
            rois = os.listdir(PATH_TUPAC_RAW_IMAGES + i_fold + '/' + i_case + '/')
            rois = [ID for ID in rois if ID != 'Thumbs.db']

            for i_roi in rois:
                im = np.array(Image.open(PATH_TUPAC_RAW_IMAGES + i_fold + '/' + i_case + '/' + i_roi))

                mask = np.zeros((im.shape[0], im.shape[1]))
                if os.path.isfile(PATH_TUPAC_RAW_GT + i_case + '/' + i_roi[:-4] + '.csv'):
                    mitosis = pd.read_csv(PATH_TUPAC_RAW_GT + i_case + '/' + i_roi[:-4] + '.csv', header=None).values

                    for i_mitosis in np.arange(0, mitosis.shape[0]):
                        mask[mitosis[i_mitosis, 0] - 1, mitosis[i_mitosis, 1] - 1] = 1

                x = 0
                y = 0
                while x + patch_size < im.shape[1]:
                    y = 0
                    while y + patch_size < im.shape[0]:
                        id = i_case + '_' + i_roi[:-4] + '_' + str(x) + '_' + str(y)

                        patch = im[y:y + patch_size, x:x + patch_size, :]
                        patch_mask = mask[y:y + patch_size, x:x + patch_size]

                        if not os.path.isfile(PATH_TUPAC_PROCESSED + 'images/' + id + '.png'):
                            Image.fromarray(patch).save(PATH_TUPAC_PROCESSED + 'images/' + id + '.png')

                        if np.max(patch_mask) == 1:
                            if not os.path.isfile(PATH_TUPAC_PROCESSED + 'masks/' + id + '.png'):
                                Image.fromarray(patch_mask * 255).convert("L").save(
                                    PATH_TUPAC_PROCESSED + 'masks/' + id + '.png')

                        y += patch_size
                    x += patch_size


def patch_midog(patch_size=500, filter_hard_positives=False):

    slides = sorted(os.listdir(PATH_MIDOG21_RAW_IMAGES))[0:150]
    gt = json.load(open(PATH_MIDOG21_RAW_GT))
    gt_table = pd.DataFrame(gt["annotations"])

    if filter_hard_positives:
        gt_table = gt_table[gt_table["category_id"] == 1]
        id_masks = "masks/"
    else:
        id_masks = "masks_hn/"

    if not os.path.exists(PATH_MIDOG21_PROCESSED + 'images/'):
        os.makedirs(PATH_MIDOG21_PROCESSED + 'images/')
    if not os.path.exists(PATH_MIDOG21_PROCESSED + id_masks):
        os.makedirs(PATH_MIDOG21_PROCESSED + id_masks)

    c = 0
    for i_slide in slides:
        print("Case " + str(c+1) + "/" + str(len(slides)), end='\n')
        c += 1

        # Get sample id
        slide_id = int(i_slide.split(".")[0])

        # Get annotated mitosis from the target slide
        gt_sample = gt_table[gt_table["image_id"] == slide_id]

        # Load image
        im = np.array(Image.open(PATH_MIDOG21_RAW_IMAGES + i_slide))

        # Create mask and asign labels
        mask = np.zeros((im.shape[0], im.shape[1]))

        # Insert annotations into mask
        for i_annotation in range(len(gt_sample)):
            bbox = gt_sample["bbox"].values[i_annotation]
            mask[int(np.mean([bbox[1], bbox[3]])), int(np.mean([bbox[0], bbox[2]]))] = 1

        # Extract patches
        x, y = 0, 0
        while x + patch_size < im.shape[1]:
            y = 0
            while y + patch_size < im.shape[0]:
                id = str(slide_id) + '_0_' + str(x) + '_' + str(y)

                patch = im[y:y + patch_size, x:x + patch_size, :]
                patch_mask = mask[y:y + patch_size, x:x + patch_size]

                if not os.path.isfile(PATH_MIDOG21_PROCESSED + 'images/' + id + '.png'):
                    Image.fromarray(patch).save(PATH_MIDOG21_PROCESSED + 'images/' + id + '.png')

                if np.max(patch_mask) == 1:
                    if not os.path.isfile(PATH_MIDOG21_PROCESSED + id_masks + id + '.png'):
                        Image.fromarray(patch_mask * 255).convert("L").save(PATH_MIDOG21_PROCESSED + id_masks + id + '.png')
                y += patch_size
            x += patch_size


def patch_CCMCT(patch_size=500):
    import openslide
    from skimage.filters import threshold_otsu

    slides = sorted(os.listdir(PATH_CCMCT_RAW_IMAGES))[0:150]
    gt = json.load(open(PATH_CCMCT_RAW_GT))
    gt_table = pd.DataFrame(gt["annotations"])

    # Get only mitotic figures
    gt_table = gt_table[gt_table["category_id"] == 2]
    id_masks = "masks/"

    # Insert slide id
    gt_table["slides"] = [CCMCT_wsi2id_lookup_dict[i] for i in list(np.int16(gt_table["image_id"].values))]

    if not os.path.exists(PATH_CCMCT_PROCESSED + 'images/'):
        os.makedirs(PATH_CCMCT_PROCESSED + 'images/')
    if not os.path.exists(PATH_CCMCT_PROCESSED + id_masks):
        os.makedirs(PATH_CCMCT_PROCESSED + id_masks)

    c = 0
    for i_slide in slides:
        print("Case " + str(c+1) + "/" + str(len(slides)), end='\n')
        c += 1

        # Get sample id
        slide_id = i_slide.split(".")[0]

        # Get annotated mitosis from the target slide
        gt_sample = gt_table[gt_table["slides"] == slide_id]

        # Openslide object from curent slide
        slide = openslide.open_slide(PATH_CCMCT_RAW_IMAGES + i_slide)

        # Background mask via otsu - using low magnification to get the threshold
        im_intensity = np.mean(np.array(slide.read_region(level=slide.level_count - 1,
                                                          location=(0, 0),
                                                          size=slide.level_dimensions[slide.level_count - 1]).convert("RGB")), -1)
        thresh = threshold_otsu(im_intensity)

        # Create mask and asign labels
        mask = np.zeros((slide.level_dimensions[0][1], slide.level_dimensions[0][0]), dtype=np.int8)

        # Insert annotations into mask
        for i_annotation in range(len(gt_sample)):
            bbox = gt_sample["bbox"].values[i_annotation]
            mask[int(np.mean([bbox[1], bbox[3]])), int(np.mean([bbox[0], bbox[2]]))] = 1

        # Extract patches
        x, y = 0, 0
        while x + patch_size < slide.level_dimensions[0][0]:
            y = 0
            while y + patch_size < slide.level_dimensions[0][1]:
                print("y: " + str(y) + "/" + str(slide.level_dimensions[0][1]) +
                      " | x: " + str(x) + "/" + str(slide.level_dimensions[0][0]), end="\r")
                id = str(slide_id) + '_0_' + str(x) + '_' + str(y)

                patch = np.array(slide.read_region(level=0, location=(x, y),
                                                   size=(patch_size, patch_size)).convert("RGB"))
                tissue_mask = - np.float32(patch.mean(-1) > thresh) + 1

                if tissue_mask.mean() <= 0.2:
                    y += patch_size
                    continue

                patch_mask = mask[y:y + patch_size, x:x + patch_size]

                if not os.path.isfile(PATH_CCMCT_PROCESSED + 'images/' + id + '.png'):
                    Image.fromarray(patch).save(PATH_CCMCT_PROCESSED + 'images/' + id + '.png')

                if np.max(patch_mask) == 1:
                    if not os.path.isfile(PATH_CCMCT_PROCESSED + id_masks + id + '.png'):
                        Image.fromarray(patch_mask * 255).convert("L").save(
                            PATH_CCMCT_PROCESSED + id_masks + id + '.png')
                y += patch_size
            x += patch_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories and partition
    parser.add_argument("--dataset", default='TUPAC16', type=str, help=" TUPAC16 | MIDOG21 | CCMCT | ")
    parser.add_argument('--extract_patches', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--stain_norm', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--filter_hard_positives', default=True, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()

    if args.dataset == "TUPAC16":
        PATH_IMAGES, PATH_IMAGES_NORM = PATH_TUPAC_PROCESSED + 'images/', PATH_TUPAC_PROCESSED + 'images_norm/'
        patch = patch_tupac
        args_patching = [500]
    elif args.dataset == "MIDOG21":
        PATH_IMAGES, PATH_IMAGES_NORM = PATH_MIDOG21_PROCESSED + 'images/', PATH_MIDOG21_PROCESSED + 'images_norm/'
        patch = patch_midog
        args_patching = [500, args.filter_hard_positives]
    elif args.dataset == "CCMCT":
        PATH_IMAGES, PATH_IMAGES_NORM = PATH_CCMCT_PROCESSED + 'images/', PATH_CCMCT_PROCESSED + 'images_norm/'
        patch = patch_CCMCT()
        args_patching = [500]
    else:
        print("Dataset not supported... ")
        PATH_IMAGES, PATH_IMAGES_NORM, patch, args_patching = None, None, None, None
        args.extract_patches = False
        args.stain_norm = False

    if args.extract_patches:
        print("Extracting patches from histology images... ", end="\n")
        patch(*args_patching)

    if args.stain_norm:
        print("Applying stain normalization to the patches... ", end="\n")
        if not os.path.exists(PATH_IMAGES_NORM):
            os.makedirs(PATH_IMAGES_NORM)
        # Set stain normalization function and reference image
        color_normalization_function = ColourNormalization(PATH_NORM_IMAGE)
        # Get images to normalize and loop over them
        files, c = sorted(os.listdir(PATH_IMAGES)), 0
        for iFile in files:
            c += 1
            print(str(c) + '/' + str(len(files)), end='\r')
            if not os.path.isfile(PATH_IMAGES_NORM + iFile.split('/')[-1]):
                im = np.array(io.imread(PATH_IMAGES + iFile))
                im = color_normalization_function(im)
                io.imsave(PATH_IMAGES_NORM + iFile.split('/')[-1], np.uint8(im))



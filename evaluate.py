import os
import torch
import numpy as np
import json
import imutils
import argparse

from skimage import io
from utils.misc import sigmoid
from utils.evaluation import evaluate_motisis_localization
from modeling.models import Resnet

from data.constants import *

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate(dir_images, cases, dir_model, patch_size=500, name_out="test", save_visualization=False,
             batch_norm_adaptation=False):

    # Read json to get threshold
    with open(dir_model + 'setting.txt', 'r') as f:
        setting = json.load(f)

    # Load model
    weights = torch.load(dir_model + "network_weights_best.pth")
    if not "backbone" in list(setting.keys()):
        setting['backbone'] = "RN18"
        for key in list(weights.keys()):
            weights[key.replace('resnet18_model', 'model')] = weights.pop(key)
    model = Resnet(in_channels=3, n_classes=1, n_blocks=setting['n_blocks'], pretrained=True,
                   mode=setting['mode'], aggregation=setting['aggregation'], backbone=setting['backbone']).to(device)
    model.load_state_dict(weights, strict=True)

    # Define weather batch norm layers are adapted during inference or not - it helps if you do not use stain norm.
    if batch_norm_adaptation:
        model.train()
    else:
        model.eval()

    TP, FP, FN = [], [], []
    for case in cases:
        print('Case: ' + case, end='\n')

        # Load images
        files = os.listdir(dir_images + 'images/')  # Check images in folder
        files_case = [iFile for iFile in files if iFile.split('_')[0] == case]   # Select images from desired case
        regions_case = np.unique([iCase.split('_')[1] for iCase in files_case])  # Select number of regions

        for iRegion in regions_case:

            images_region = [iFile for iFile in files_case if iFile.split('_')[1] == iRegion]

            images_all = []
            images_norm_all = []
            mask_all = []
            mask_pred_all = []

            for i_image in images_region:

                # Original image
                im = np.array(io.imread(dir_images + 'images/' + i_image))
                im = imutils.resize(im, height=patch_size)
                im = np.transpose(im, (2, 0, 1)) / 255

                # Color-normalized image
                im_norm = np.array(io.imread(dir_images + 'images_norm/' + i_image))
                im_norm = imutils.resize(im_norm, height=patch_size)
                im_norm = np.transpose(im_norm, (2, 0, 1)) / 255

                # Mask
                if os.path.isfile(dir_images + 'masks/' + i_image):
                    mask = np.array(io.imread(dir_images + 'masks/' + i_image))
                    mask = imutils.resize(mask, height=patch_size)
                    mask = mask / 255
                    mask = np.double(mask > 0)
                else:
                    mask = np.zeros((patch_size, patch_size))

                # Get localization mask
                with torch.no_grad():
                    pred_logits, cam_logits = model(torch.tensor(im_norm).float().unsqueeze(0).to(device))
                    mask_pred = sigmoid(cam_logits.squeeze().cpu().detach().numpy())

                images_all.append(im)
                images_norm_all.append(im_norm)
                mask_all.append(mask)
                mask_pred_all.append(mask_pred)

            # Reconstruct whole region
            nCols = np.max([int(i_image.split('_')[2]) for i_image in images_region]) + patch_size
            nRows = np.max([int(i_image.split('_')[3][0:-4]) for i_image in images_region]) + patch_size

            whole_image = np.zeros((3, nRows, nCols))
            whole_mask = np.zeros((nRows, nCols))
            whole_mask_pred = np.zeros((nRows, nCols))
            for iCol in np.arange(0, nCols, patch_size, dtype=int):
                for iRow in np.arange(0, nRows, patch_size, dtype=int):
                    idx = images_region.index(case + '_' + iRegion + '_' + str(iCol) + '_' + str(iRow) + '.png')

                    whole_image[:, iRow:iRow+patch_size, iCol:iCol+patch_size] = images_all[idx]
                    whole_mask[iRow:iRow + patch_size, iCol:iCol + patch_size] = mask_all[idx]
                    whole_mask_pred[iRow:iRow + patch_size, iCol:iCol + patch_size] = mask_pred_all[idx]

            # Visualize
            loc_test_metrics = evaluate_motisis_localization(X=np.expand_dims(whole_image, 0),
                                                             M=np.expand_dims(whole_mask, 0),
                                                             Mhat=np.expand_dims(whole_mask_pred, 0),
                                                             ids=[case + '_' + iRegion + '.png'],
                                                             dir_out=dir_model + '/visualizations_region_' + name_out + '/',
                                                             th=setting["threshold_val"],
                                                             save_visualization=save_visualization)

            TP.append(loc_test_metrics['TP'])
            FP.append(loc_test_metrics['FP'])
            FN.append(loc_test_metrics['FN'])

    TP = np.sum(TP)
    FP = np.sum(FP)
    FN = np.sum(FN)

    precision = TP / (TP + FP + 1e-3)
    recall = TP / (TP + FN + 1e-3)
    F1 = (2 * recall * precision) / (recall + precision + 1e-3)

    print("Results: F1=%2.3f | Recall=%2.3f | Precision=%2.3f | " % (F1, recall, precision), end="\n")
    print("Disentangled: TP=%d | FP=%d | FN=%d | " % (TP, FP, FN), end="\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories and partition
    parser.add_argument("--dataset", default='TUPAC16', type=str, help=" TUPAC16 / MIDOG21 ")
    parser.add_argument("--dir_model", default='./local_data/results/TUPAC16_UTS_teacher_weakAugm_locCons/',
                        type=str)
    parser.add_argument('--save_visualization', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--batch_norm_adaptation', default=False, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()

    # Get evaluation setting
    if args.dataset == "TUPAC16":
        dir_images = PATH_TUPAC_PROCESSED
        cases = TUPAC16_ID_TRAIN
        dataset_implemented = True
    elif args.dataset == "MIDOG21":
        dir_images = PATH_MIDOG21_PROCESSED
        cases = MIDOG21_ID_TEST
        dataset_implemented = True
    else:
        print("Specified dataset not implemented for evaluation...", end="\n")
        dataset_implemented, dir_images, cases = False, None, None

    # Evauate
    if dataset_implemented:
        print("Evaluation mitosis localization at the slide level...", end="\n")
        evaluate(dir_images, cases, args.dir_model, patch_size=500, name_out="test_" + args.dataset,
                 save_visualization=args.save_visualization, batch_norm_adaptation=args.batch_norm_adaptation)

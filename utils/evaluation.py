import os
import PIL

import numpy as np

from skimage import measure
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, confusion_matrix,\
    precision_score, recall_score

from utils.misc import distance_mitosis


def evaluate_image_level(refs, preds):
    auc = roc_auc_score(refs > 0.5, preds)

    # Calculate pr curve and its area
    precision, recall, threshold = precision_recall_curve(refs > 0.5, preds)

    # Search the optimum point and obtain threshold via f1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    f1[np.isnan(f1)] = 0
    th = threshold[np.argmax(f1)]

    accuracy = accuracy_score(refs > th, preds >= th)
    prec = precision_score(refs > th, preds >= th)
    rec = recall_score(refs > th, preds >= th)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-6)

    cm = confusion_matrix(refs > 0.5, preds >= th)

    return {'accuracy': accuracy, 'f1': f1, 'AUC': auc, 'cm': cm}, th


def evaluate_motisis_localization(X, M, Mhat, ids, dir_out="", th=0.5, save_visualization=False, maximum_d=30.,
                                  visualize_negatives=False):

    # Prepare folders
    if save_visualization:
        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)
        if not os.path.isdir(dir_out + 'positives/'):
            os.mkdir(dir_out + 'positives/')
        if not os.path.isdir(dir_out + 'negatives/'):
            os.mkdir(dir_out + 'negatives/')

    TP, FP, FN = 0., 0., 0.
    for i_case in np.arange(0, X.shape[0]):
        print(str(i_case + 1) + '/' + str(X.shape[0]), end='\r')

        TP_case = []
        FP_case = []
        FN_case = []

        x = np.transpose(X[i_case, :, :, :], (1, 2, 0))

        img = PIL.Image.fromarray((x * 255).astype(np.uint8))
        draw = PIL.ImageDraw.Draw(img)

        m = M[i_case, :, :]
        mhat = Mhat[i_case, :, :] > th

        # labelled mitosis
        labels_ref = measure.label(m)
        props_ref = measure.regionprops(labels_ref)

        # Predicted mitosis
        labels_pred = measure.label(mhat)
        props_pred = measure.regionprops(labels_pred)

        # Check for FN
        if len(props_ref) > 0:
            for iprop in props_ref:
                if len(props_pred) == 0:
                    FN += 1
                    FN_case.append(iprop.centroid)
                else:
                    d_pred = [distance_mitosis(np.array(i_prop_pred.centroid), np.array(iprop.centroid)) for i_prop_pred in props_pred]
                    if np.min(d_pred) > maximum_d:
                        FN += 1
                        FN_case.append(iprop.centroid)

        # Check for TP and FP
        if len(props_pred) > 0:
            for iprop in props_pred:
                if len(props_ref) == 0:
                    FP += 1
                    FP_case.append(iprop.centroid)
                else:
                    d_ref = [distance_mitosis(np.array(i_prop_ref.centroid), np.array(iprop.centroid)) for i_prop_ref in props_ref]
                    if np.min(d_ref) < maximum_d:
                        TP += 1
                        TP_case.append(iprop.centroid)
                    else:
                        FP += 1
                        FP_case.append(iprop.centroid)

        if save_visualization:

            r = 40
            w = 10
            for icentroid in FN_case:
                color = (0, 0, 255)
                draw.ellipse((icentroid[1] - r, icentroid[0] - r, icentroid[1] + r, icentroid[0] + r),
                             outline=color, fill=None, width=w)
            for icentroid in TP_case:
                color = (0, 255, 0)
                draw.ellipse((icentroid[1] - r, icentroid[0] - r, icentroid[1] + r, icentroid[0] + r),
                             outline=color, fill=None, width=w)
            for icentroid in FP_case:
                color = (255, 255, 0)

                draw.ellipse((icentroid[1] - r, icentroid[0] - r, icentroid[1] + r, icentroid[0] + r),
                             outline=color, fill=None, width=w)

            if len(FN_case) == 0 and len(TP_case) == 0 and len(FP_case) == 0:
                if visualize_negatives:
                    img.save(dir_out + 'negatives/' + ids[i_case])
            else:
                img.save(dir_out + 'positives/' + ids[i_case])

    precision = TP / (TP + FP + 1e-3)
    recall = TP / (TP + FN + 1e-3)

    F1 = (2*recall*precision)/(recall+precision+1e-3)

    return {'TP': TP, 'FP': FP, 'FN': FN, 'F1': F1}
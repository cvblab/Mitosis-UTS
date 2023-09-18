import torch
import json

import numpy as np

from utils.misc import sigmoid, centroid_from_masks
from utils.evaluation import evaluate_image_level, evaluate_motisis_localization
from modeling.models import Resnet
from modeling.losses import log_barrier

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def predict_dataset(dataset, model, bs=32):
    model.eval()

    iSample = 0
    Yhat, Mhat = [], []
    X, Y = dataset.X, dataset.Y
    while iSample < X.shape[0]:
        print(str(iSample + 1) + '/' + str(X.shape[0]), end='\r')

        batch = X[iSample:iSample+bs, :, :, :]
        batch = torch.tensor(batch).float().to(device)

        if bs == 1:
            batch.unsqueeze(0)

        pred_logits, cam_logits = model(batch)
        yhat = pred_logits

        yhat = yhat.unsqueeze(-1).cpu().detach().numpy()
        cam_logits = cam_logits.cpu().detach().numpy()

        if len(yhat.shape) == 1:  # Security check for the case when there is only one sample in the batch
            yhat = np.expand_dims(yhat, -1)

        Yhat.append(yhat)
        Mhat.append(cam_logits)

        iSample += bs

    Yhat = np.squeeze(np.concatenate(Yhat))
    Mhat = np.squeeze(np.concatenate(Mhat, 0))
    return Y, Yhat, Mhat


def distill_dataset(train_dataset, teacher_model_path, distance_distillation=True, pseudolabel=False,
                    filter_criteria="FP_TP"):

    # Read json to get threshold
    with open(teacher_model_path + 'setting.txt', 'r') as f:
        setting = json.load(f)

    # Load model
    weights = torch.load(teacher_model_path + "network_weights_best.pth")
    if not "backbone" in list(setting.keys()):
        setting['backbone'] = "RN18"
        for key in list(weights.keys()):
            weights[key.replace('resnet18_model', 'model')] = weights.pop(key)
    model_t = Resnet(in_channels=3, n_classes=1, n_blocks=setting['n_blocks'], pretrained=True,
                   mode=setting['mode'], aggregation=setting['aggregation'], backbone=setting['backbone']).to(device)
    model_t.load_state_dict(weights)

    # Predictions on training dataset
    Y, Yhat, Mhat = predict_dataset(train_dataset, model_t, bs=32)
    model_t.cpu()

    # Obtain threshold for mitosis detection
    _, th_val = evaluate_image_level(Y, sigmoid(Yhat))

    # Obtain weak, hard pseudo-labels predicted by the model
    Ypseu = np.expand_dims(np.float32(np.array(sigmoid(Yhat) > th_val)), -1)

    if pseudolabel:
        train_dataset.Y = Ypseu
        return train_dataset

    # 1st distillation: remove samples with discrepancy at the global level
    idx = np.squeeze(np.argwhere((Y[:, 0] - Ypseu[:, 0]) == 0))
    train_dataset.filter_cases(idx)
    Mhat = Mhat[idx, :, :]

    if distance_distillation:

        # Training samples with at least 1 mitosis
        idx = np.squeeze(np.argwhere(np.squeeze(train_dataset.N) >= 1))

        # Loop over each image with a positive mitosis
        TPs = []
        for i_index in idx:

            # Evaluate the mitosis localized on that sample
            metrics = evaluate_motisis_localization(
                X=np.expand_dims(train_dataset.X[i_index, :, :, :], 0),
                M=np.expand_dims(train_dataset.M[i_index, :, :], 0),
                Mhat=np.expand_dims(sigmoid(Mhat[i_index, :, :]), 0),
                ids=train_dataset.images, th=th_val, save_visualization=False)

            # Check exclusion criterias for the sample
            if filter_criteria == 'TP':  # Criteria: if there is at least one good detected mitosis, the sample passes
                TPs.append(metrics['TP'] == 0)
            elif filter_criteria == 'FP':  # Criteria: if exists any false positive, the sample not passes
                TPs.append(metrics['FP'] > 0)
            elif filter_criteria == 'FP_TP':  # Only perfect cases pass
                TPs.append(metrics['FP'] > 0 or metrics['TP'] == 0)

        # Obtain eliminated indexes
        idx2 = np.squeeze(np.argwhere(TPs))
        if isinstance(idx2.tolist(), (int, np.integer)):
            idx2 =[idx2.tolist()]

        idx3 = np.array([i for i in range(train_dataset.images.__len__()) if i not in idx[idx2]])
        # Filter dataset
        train_dataset.filter_cases(idx3)

    return train_dataset


def constraint_localization(Y, M, cam_logits, constraint_type="l2", margin=0.0, tlb=5, temperature=10, input_shape=500):

    # Select cases with mitosis
    idx = np.argwhere(Y.detach().cpu().numpy() == 1)

    # At least one case must contain mitoses
    if len(list(idx)) > 0:

        # Get GT
        M_mitosis = M[idx, :, :]
        C = centroid_from_masks(M_mitosis > 0)
        C = torch.tensor(C).cuda()

        # Compute centroids
        Mhat_mitosis = torch.index_select(cam_logits, 0, torch.tensor(idx).cuda().squeeze()).squeeze()
        Mhat_mitosis = torch.sigmoid(Mhat_mitosis * temperature)

        grid_x = torch.tensor(np.arange(0, input_shape)).cuda().unsqueeze(0)
        grid_y = torch.tensor(np.arange(0, input_shape)).cuda().unsqueeze(0)

        p_x = (Mhat_mitosis.sum(-2) * grid_x).sum(-1) / Mhat_mitosis.sum(-2).sum(-1)
        p_y = (Mhat_mitosis.sum(-1) * grid_y).sum(-1) / Mhat_mitosis.sum(-1).sum(-1)

        ((Mhat_mitosis.sum(-2) * grid_x + 1e-3) / Mhat_mitosis.sum()).mean() * 512
        ((Mhat_mitosis.sum(-1) * grid_x + 1e-3) / Mhat_mitosis.sum()).mean() * 512

        Cpred = torch.cat([p_y.unsqueeze(-1), p_x.unsqueeze(-1)], dim=-1)

        # Compute distance loss
        d = torch.sqrt(torch.sum(torch.square(C - Cpred), dim=-1))

        # Update overall losses
        if constraint_type == 'l2':
            constraint_loss = torch.mean(d)
        elif constraint_type == 'lb':
            constraint_loss = log_barrier((d.unsqueeze(-1) - margin), t=tlb) / d.shape[0]
        else:
            constraint_loss = torch.mean(d)

    else:
        constraint_loss, d = torch.tensor(0), 0

    return constraint_loss, d
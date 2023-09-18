import argparse
import json
import torch
import os

import pandas as pd
import numpy as np

from data.dataset import Dataset, Generator
from data.constants import *
from modeling.models import Resnet
from modeling.utils import distill_dataset, predict_dataset, constraint_localization
from utils.misc import set_seeds, sigmoid
from utils.evaluation import evaluate_image_level, evaluate_motisis_localization

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# set_seeds(42, torch.cuda.is_available())


def main(args):

    # Prepare folders
    if not os.path.isdir(PATH_RESULTS + args.experiment_name + '/'):
        os.makedirs(PATH_RESULTS + args.experiment_name + '/')

    # Set train dataset
    train_dataset = Dataset(args.dataset_id, partition='train', input_shape=args.input_shape, labels=1,
                            preallocate=args.preallocate, stain_normalization=False, dir_images=args.dir_images,
                            only_one_mitosis=True, dir_masks=args.dir_masks)
    # Set validation and testing datasets
    val_dataset = Dataset(args.dataset_id, partition='val', input_shape=args.input_shape, labels=1,
                          preallocate=args.preallocate, dir_images=args.dir_images, dir_masks=args.dir_masks)
    test_dataset = Dataset(args.dataset_id, partition='test', input_shape=args.input_shape, labels=1,
                           preallocate=args.preallocate, dir_images=args.dir_images, dir_masks="masks")

    # Training dataset distillation - only for student training
    if args.distillation:
        teacher_model_path = PATH_RESULTS + args.teacher_experiment + "/"
        if os.path.isfile(teacher_model_path + 'network_weights_best.pth'):
            print("Distilling training/validation dataset using " + args.teacher_experiment + " model!")
            train_dataset = distill_dataset(train_dataset, teacher_model_path,
                                            distance_distillation=args.distillation,
                                            filter_criteria=args.filter_criteria, pseudolabel=args.pseudolabel)
            val_dataset = distill_dataset(val_dataset, teacher_model_path,
                                          distance_distillation=args.distillation,
                                          filter_criteria=args.filter_criteria, pseudolabel=args.pseudolabel)
        else:
            print("Unavailable model for distillation: " + args.teacher_experiment + " ... not training")
            return

    # Prepare data generator
    train_generator = Generator(train_dataset, args.batch_size, shuffle=True, balance=True,
                                strong_augmentation=args.strong_augmentation)

    # Network architecture
    model = Resnet(in_channels=args.input_shape[0], n_classes=1, n_blocks=args.n_blocks, pretrained=args.pretrained,
                   mode=args.mode, aggregation=args.aggregation, backbone=args.backbone).to(device)
    # Set losses
    Lce = torch.nn.BCEWithLogitsLoss().to(device)
    # Set optimizer
    opt = torch.optim.Adam(list(model.parameters()), lr=args.lr)

    # Training loop
    history, val_acc_min = [], 0
    for i_epoch in range(args.epochs):

        Y_train , Yhat_train = [], []
        loss_ce_over_all, loss_constraint_over_all = 0.0, 0.0
        for i_iteration, (X, Y, M) in enumerate(train_generator):
            model.train()

            X = torch.tensor(X).cuda().float().to(device)
            Y = torch.tensor(Y).cuda().float().to(device)

            # Apply data augmentation
            if args.augmentation:
                # Forward augmentation
                M = torch.tensor(M).cuda().float().to(device)
                X, M = train_generator.augmentations(X, M.unsqueeze(1))
                # Check if the mitosis is still in the image after spatial augmentations
                Y = M.contiguous().view((M.shape[0], -1)).max(-1)[-0].detach()  # New mitosis label
                Y = (Y > 0).to(torch.float32)
                M = torch.squeeze(M).cpu().detach().numpy()  # Recover mask as array

            # Forward network
            pred_logits, cam_logits = model(X, reshape_cam=True)

            # Estimate losses
            ce = Lce(pred_logits, torch.squeeze(Y))
            L = ce * args.alpha_ce

            if args.location_constraint:
                constraint_loss, d = constraint_localization(Y, M, cam_logits, constraint_type=args.constraint_type,
                                                             margin=args.margin, tlb=args.tlb,
                                                             temperature=args.temperature,
                                                             input_shape=args.input_shape[-1])
                L += constraint_loss * args.alpha_location

            # Backward gradients
            L.backward()
            opt.step()
            opt.zero_grad()

            # Track predictions and losses
            Y_train.append(Y.detach().cpu().numpy())
            Yhat_train.append(torch.sigmoid(pred_logits).detach().cpu().numpy())
            loss_ce_over_all += ce.item()
            if args.location_constraint:
                loss_constraint_over_all += torch.mean(d).item()

            # Display losses and acc per iteration
            info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f}".format(
                    i_epoch + 1, args.epochs, i_iteration + 1, len(train_generator), ce.detach().cpu().numpy())
            if args.location_constraint:
                info += " || Lcons={:.4f}".format(torch.mean(d).detach().cpu().numpy())
            print(info, end='\r')

        # --- Validation at epoch end
        model.eval()

        # Validation predictions
        Y_val, Yhat_val, Mhat_val = predict_dataset(val_dataset, model, bs=32)
        # Test predictions
        Y_test, Yhat_test, Mhat_test = predict_dataset(test_dataset, model, bs=32)

        # Train metrics
        metrics_train, _ = evaluate_image_level(np.concatenate(Y_train), np.concatenate(Yhat_train, 0))
        loss_training = loss_ce_over_all / len(train_generator)
        # Validation metrics
        metrics_val, th = evaluate_image_level(Y_val, sigmoid(Yhat_val))
        loss_val = Lce(torch.tensor(Yhat_val).cuda().unsqueeze(-1), torch.tensor(Y_val).cuda()).detach().cpu().numpy()
        # Test metrics
        metrics_test, _ = evaluate_image_level(Y_test, sigmoid(Yhat_test))

        # Display losses per epoch
        info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f} accuracy={:.4f} ; f1={:.4f} || Lce_val={:.4f} accuracy_val={:.4f} ; f1={:.4f}".format(
            i_epoch + 1, args.epochs, len(train_generator), len(train_generator), loss_training, metrics_train["accuracy"], metrics_train["f1"],
            loss_val, metrics_val["accuracy"], metrics_val["f1"])
        if args.location_constraint:
            loss_constraint_training = loss_constraint_over_all / len(train_generator)
            info += " || Lcons={:.4f}".format(loss_constraint_training)
        print(info, end='\n')

        # Track learning curves
        h = [loss_training, loss_val, metrics_train["accuracy"], metrics_train["f1"], metrics_val["accuracy"],
             metrics_val["f1"], metrics_test["accuracy"], metrics_test["f1"]]
        h_caption = ['loss_train', 'loss_val', 'metric_train_acc', 'metric_train_f1', 'metric_val_acc',
                     'metric_val_f1', 'metric_test_acc', 'metric_test_f1']
        if args.location_constraint:
            h.append(loss_constraint_training)
            h_caption.append('loss_centroid')

        # Save learning curves
        history.append(h)
        history_final = pd.DataFrame(history, columns=h_caption)
        history_final.to_excel(PATH_RESULTS + args.experiment_name + '/lc.xlsx')

        # Save model
        if metrics_val["f1"] > val_acc_min:
            print('Validation F1 improved from ' + str(round(val_acc_min, 5)) + ' to ' + str(
                round(metrics_val["f1"], 5)) + '  ... saving model')
            torch.save(model.state_dict(), PATH_RESULTS + args.experiment_name + '/network_weights_best.pth')
            val_acc_min = metrics_val["f1"]
        print('Validation cm: ', end='\n')
        print(metrics_val["cm"], end='\n')
        print('Test cm: ', end='\n')
        print(metrics_test["cm"], end='\n')

    # Save last model
    torch.save(model.state_dict(), PATH_RESULTS + args.experiment_name + '/network_weights_last.pth')

    # Final localization results using the best model
    model.load_state_dict(torch.load(PATH_RESULTS + args.experiment_name + '/network_weights_best.pth'))

    print('Predicting to validate', end='\n')
    # Validation predictions
    Y_val, Yhat_val, Mhat_val = predict_dataset(val_dataset, model, bs=32)
    # Validation metrics
    metrics_val, th_val = evaluate_image_level(Y_val, sigmoid(Yhat_val))
    # Test predictions
    Y_test, Yhat_test, Mhat_test = predict_dataset(test_dataset, model, bs=32)
    # Test metrics
    metrics_test, th_test = evaluate_image_level(Y_test, sigmoid(Yhat_test))

    # Validation localization
    print('Localization evaluation on Validation: ', end='\n')
    loc_val_metrics = evaluate_motisis_localization(val_dataset.X, val_dataset.M, sigmoid(Mhat_val),
                                                    val_dataset.images,
                                                    PATH_RESULTS + args.experiment_name + '/visualizations_val/',
                                                    th=th_val, save_visualization=args.save_visualization)
    loc_val_metrics['th'] = float(round(th_val, 2))
    with open(PATH_RESULTS + args.experiment_name + '/localization_val.txt', 'w') as file:
        file.write(json.dumps(loc_val_metrics))
    print("Results: F1=%2.3f |" % (loc_val_metrics["F1"]), end="\n")
    print("Disentangled: TP=%d | FP=%d | FN=%d | " % (loc_val_metrics["TP"], loc_val_metrics["FP"],
                                                      loc_val_metrics["FN"]), end="\n")

    # Test localization
    print('Localization evaluation on Test: ', end='\n')
    loc_test_metrics = evaluate_motisis_localization(test_dataset.X, test_dataset.M, sigmoid(Mhat_test),
                                                     test_dataset.images,
                                                     PATH_RESULTS + args.experiment_name + '/visualizations_test/',
                                                     th=th_val, save_visualization=args.save_visualization)
    loc_test_metrics['th'] = float(round(th_test, 2))
    with open(PATH_RESULTS + args.experiment_name + '/localization_test.txt', 'w') as file:
        file.write(json.dumps(loc_test_metrics))
    print("Results: F1=%2.3f |" % (loc_test_metrics["F1"]), end="\n")
    print("Disentangled: TP=%d | FP=%d | FN=%d | " % (loc_test_metrics["TP"], loc_test_metrics["FP"],
                                                      loc_test_metrics["FN"]), end="\n")

    # Save input args
    args.threshold_val, args.threshold_test = float(round(th_val, 2)), float(round(th_test, 2))
    argparse_dict = vars(args)
    with open(PATH_RESULTS + args.experiment_name + '/setting.txt', 'w') as f:
        json.dump(argparse_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories and partition
    parser.add_argument("--dataset_id", default='TUPAC16', type=str)
    parser.add_argument("--dir_results", default='./local_data/results/', type=str)
    parser.add_argument("--dir_images", default='images_norm', type=str)
    parser.add_argument("--dir_masks", default='masks', type=str)
    parser.add_argument("--experiment_name", default="student_TUPAC", type=str)

    # Hyperparameters
    parser.add_argument("--input_shape", default=(3, 500, 500), type=list)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--n_blocks", default=3, type=int)
    parser.add_argument("--pretrained", default=True, type=lambda x: (str(x).lower() == 'true'), help="xxx")
    parser.add_argument("--augmentation", default=True, type=lambda x: (str(x).lower() == 'true'), help="xxx")
    parser.add_argument("--strong_augmentation", default=False, type=lambda x: (str(x).lower() == 'true'), help="xxx")
    parser.add_argument("--backbone", default="RN18", type=str)

    # Weakly supervised setting
    parser.add_argument("--mode", default='instance', type=str)
    parser.add_argument("--aggregation", default='max', type=str)

    # Constraints
    parser.add_argument("--alpha_ce", default=1, type=float)
    parser.add_argument("--location_constraint", default=False, type=lambda x: (str(x).lower() == 'true'), help="xxx")
    parser.add_argument("--alpha_location", default=0.01, type=float)
    parser.add_argument("--temperature", default=10, type=float)
    parser.add_argument("--margin", default=40, type=float)
    parser.add_argument("--tlb", default=50, type=float)
    parser.add_argument("--constraint_type", default='l2', type=str, help="Options: l2 / lp ")

    # Teacher-Student setting
    parser.add_argument("--teacher_experiment", default='TUPAC16_UTS_teacher', type=str)
    parser.add_argument("--pseudolabel", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--distillation", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--filter_criteria", default='TP', type=str)

    # Other settings
    parser.add_argument("--preallocate", default=True, type=lambda x: (str(x).lower() == 'true'), help="xxx")
    parser.add_argument("--save_visualization", default=False, type=lambda x: (str(x).lower() == 'true'), help="xxx")

    args = parser.parse_args()
    main(args)

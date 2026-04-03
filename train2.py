import argparse
from gc import unfreeze
import os
import random
from typing import Sequence
import numpy as np
from tqdm import tqdm
from learning_rate_helper.learning_rate import PiecewiseScheduler
from loss_helper.TripletMarginLoss import TripletMarginLoss
from model_builder.model import Model
from dataset_helper.DatasetLoaderV3 import SiameseFingerprintDataset
from utils.MetricCalV3 import MetricCalV3
from utils.Utilities import Get_Min_EER, Loading_Checkpoint, Saving_Best, Saving_Checkpoint, Saving_Metric2, YAML_Reader, get_mean_std
# from CBAM_Resnet import Model as CBAM_Resnet

import torch
import torch.nn as nn
import torch.optim as optim

# from timm.loss.cross_entropy import SoftTargetCrossEntropy

def parse_args():
    parser = argparse.ArgumentParser(description="A simple argparse example")
    
    # Add arguments
    parser.add_argument(
    "--cfg",
    type=str,
    default="config/default_config.yaml",
    help="Config file used to train the model (default: config/default_config.yaml)"
    )
    args = parser.parse_args()
    config = YAML_Reader(args.cfg)
    return config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(epoch: int, end_epoch: int, model, loader, criterion, optimizer, device):
    model.train()
    # freeze_bn(model)
    metrics = MetricCalV3(device=device)
    for inputs, targets in tqdm(loader, total=len(loader), desc="Training epoch [{0}/{1}]".
                                format(epoch, end_epoch)):

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(inputs) #Output là không gian đặc trưng
        loss, frac_pos_triplets = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        metrics.update_train(loss=loss, outputs=outputs, targets=targets)
    metrics.compute_fpr_tpr_thresholds()
    return metrics

def validate(epoch, end_epoch, model, loader, criterion, device):
    model.eval()
    metrics = MetricCalV3(device=device)
    with torch.no_grad():
        for inputs, targets in tqdm(loader, total=len(loader), desc="Validating epoch [{0}/{1}]".
                                format(epoch, end_epoch)):
            inputs1 = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss, frac_pos_triplets = criterion(outputs, targets)
            metrics.update_test(loss=loss, outputs=outputs, targets=targets)
    metrics.compute_fpr_tpr_thresholds()
    return metrics

def main():
    config = parse_args()


    #Data parameters
    # root_path = config["DATASET"]["ROOT_FOLDER"]
    train_path = config["DATASET"]["TRAIN_FOLDER"]
    train_sample = int(config["DATASET"]["TRAIN_SAMPLE"])
    test_path = config["DATASET"]["TEST_FOLDER"]
    test_sample = int(config["DATASET"]["TEST_SAMPLE"])
    # CLASSES = sorted([i for i in os.listdir(root_path)])
    mean: Sequence[float] = config["TRAIN"]["DATA"]["MEAN"]
    std: Sequence[float] = config["TRAIN"]["DATA"]["STD"]
    batch_size = config["TRAIN"]["DATA"]["BATCH_SIZE"]
    

    #Training parameters
    img_size = config["TRAIN"]["DATA"]["IMAGE_SIZE"]
    enabled_transform = config["TRAIN"]["TRANSFORM"]
    begin_epoch = config["TRAIN"]["TRAIN_PARA"]["BEGIN_EPOCH"] 
    end_epoch = config["TRAIN"]["TRAIN_PARA"]["END_EPOCH"]
    resume = config["TRAIN"]["TRAIN_PARA"]["RESUME"]
    early_stopping = int(config["TRAIN"]["TRAIN_PARA"]["EARLY_STOPPING"])
    patience = config["TRAIN"]["TRAIN_PARA"]["PATIENCE"]
    epochs_no_improve = 0
    model_type = int(config["TRAIN"]["TRAIN_PARA"]["MODEL_TYPE"])
    embedding_dim = int(config["TRAIN"]["TRAIN_PARA"]["EMBEDDING_DIM"])

    #Learning_rate
    Learning_rate_para = config["TRAIN"]["LEARNING_RATE"]["PieceWise"]

    #Optional
    save_checkpoint = config["TRAIN"]["OPTIONAL"]["SAVE_CHECKPOINT"]
    save_best = config["TRAIN"]["OPTIONAL"]["SAVE_BEST"]
    save_metrics = config["TRAIN"]["OPTIONAL"]["SAVE_METRICS"]
    checkpoint_path = config["TRAIN"]["OPTIONAL"]["CHECKPOINT_PATH"]
    best_path = config["TRAIN"]["OPTIONAL"]["BEST_PATH"]
    metrics_path = config["TRAIN"]["OPTIONAL"]["METRICS_PATH"]
    
    set_seed()
    
    if mean is None or std is None:
        print("Calculating mean and std")
        mean, std = get_mean_std(train_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data = SiameseFingerprintDataset(image_size=img_size, 
                                           mean=mean, 
                                           std=std, 
                                           batch_size=batch_size, 
                                           data_path=train_path
                                        )
    test_data = SiameseFingerprintDataset(image_size=img_size, 
                                           mean=mean, 
                                           std=std, 
                                           batch_size=batch_size, 
                                           data_path=test_path
                                        )

    training_loader = train_data.build_dataset("train")
    testing_loader = test_data.build_dataset("test")

    model = Model(embedding_dim=embedding_dim).to(device)
    eval_criterion = TripletMarginLoss(margin=0.2)
    train_criterion = TripletMarginLoss(margin=0.2)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
   

    best_eer = 999999999999

    if resume == True:
        begin_epoch = Loading_Checkpoint(path=checkpoint_path,
                                         model=model,
                                         optimizer=optimizer,
                                         scheduler=None,
                                         device=device)
        best_eer = Get_Min_EER(metrics_path)
    for epoch in range(begin_epoch, end_epoch):
        train_metrics = train(epoch, 
                                end_epoch,
                                model=model, 
                                loader=training_loader, 
                                criterion=train_criterion, 
                                optimizer=optimizer, 
                                device=device)
        train_loss, train_ROC_AUC = train_metrics.avg_loss, train_metrics.ROC_AUC
        train_EER = train_metrics.EER
        train_TAR_and_FAR_1e3p = train_metrics.tar_at_far
        print()
        
        val_metrics = validate(epoch, end_epoch, model, testing_loader, eval_criterion, device)
        val_loss, val_ROC_AUC = val_metrics.avg_loss, val_metrics.ROC_AUC
        val_EER = val_metrics.EER
        val_TAR_and_FAR_1e3p = val_metrics.tar_at_far
        print()

        if save_checkpoint == True:
            Saving_Checkpoint(epoch=epoch, 
                            model=model, 
                            optimizer=optimizer, 
                            scheduler=None,
                            last_epoch=epoch, 
                            path=checkpoint_path)

        print("Epoch [{0}/{1}]: Training loss: {2}, ROC AUC: {3}, EER: {4}, TAR @ FAR=1%: {5}".
            format(epoch, end_epoch, train_loss, train_ROC_AUC, train_EER, train_TAR_and_FAR_1e3p))
        print("Epoch [{0}/{1}]: Validation loss: {2}, ROC AUC: {3}, EER: {4}, TAR @ FAR=1%: {5}".
            format(epoch, end_epoch, val_loss, val_ROC_AUC, val_EER, val_TAR_and_FAR_1e3p))
        if val_EER < best_eer:
            if save_best == True:
                print("Validation EER improve from {0} to {1} at epoch {2}. Saving best result".
                    format(round(best_eer, 4), round(val_EER, 4),  epoch))
                Saving_Best(model, best_path)
            else:
                print("Validation accuracy improve from {0} to {1} at epoch {2}".
                    format(round(best_eer, 4), round(val_EER, 4),  epoch))
            best_eer = val_EER
            epochs_no_improve = 0  # reset patience
        else:
            epochs_no_improve += 1
        if save_metrics:
            Saving_Metric2(epoch=epoch, 
                           train_loss=train_loss,
                           train_ROC_AUC=train_ROC_AUC,
                           train_EER=train_EER,
                           train_TAR_and_FAR_1e3p=train_TAR_and_FAR_1e3p,
                           val_loss=val_loss,
                           val_ROC_AUC=val_ROC_AUC,
                           val_EER=val_EER,
                           val_TAR_and_FAR_1e3p=val_TAR_and_FAR_1e3p,
                           path=metrics_path)
        if epochs_no_improve >= patience and early_stopping == True:
            print("Early stopping triggered at epoch {0}".format(epoch))
            break
        print()

    
if __name__ == '__main__':
    main()

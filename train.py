import argparse
import os
import random
from typing import Sequence
import numpy as np
from tqdm import tqdm
# from Aug.BatchWiseAug import BatchWiseAug
# from Metrics.MetricCal import MetricCal
from learning_rate_helper.learning_rate import PiecewiseScheduler
from model_builder.model import SiameseModel
from dataset_helper.DatasetLoader import DatasetLoader
from utils.MetricCalV2 import MetricCalV2
from utils.Utilities import Get_Max_Acc, Loading_Checkpoint, Saving_Best, Saving_Checkpoint, Saving_Metric2, YAML_Reader, get_mean_std
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
    metrics = MetricCalV2(device=device)
    for inputs1, inputs2, targets in tqdm(loader, total=len(loader), desc="Training epoch [{0}/{1}]".
                                format(epoch, end_epoch)):

        inputs1, inputs2 = inputs1.to(device, non_blocking=True), inputs2.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs1, outputs2 = model(inputs1, inputs2)
        loss = criterion(outputs1, outputs2, targets)
        loss.backward()
        optimizer.step()

        metrics.update_train(loss=loss, outputs1=outputs1, outputs2=outputs2, targets=targets)
    return metrics

def validate(epoch, end_epoch, model, loader, criterion, device):
    model.eval()
    metrics = MetricCalV2(device=device)
    with torch.no_grad():
        for inputs1, inputs2, targets in tqdm(loader, total=len(loader), desc="Validating epoch [{0}/{1}]".
                                format(epoch, end_epoch)):
            inputs1, inputs2 = inputs1.to(device, non_blocking=True), inputs2.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs1, outputs2 = model(inputs1, inputs2)
            loss = criterion(outputs1, outputs2, targets)
            metrics.update_test(loss=loss, outputs1=outputs1, outputs2=outputs2, targets=targets)
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
    
    train_data = DatasetLoader(path=train_path, batch_size=batch_size, number_of_sample=train_sample,std=std, mean=mean, img_size=img_size, transform=enabled_transform)
    test_data = DatasetLoader(path=test_path, batch_size=batch_size, number_of_sample=test_sample, std=std, mean=mean, img_size=img_size)

    training_loader = train_data.dataset_loader("train")
    testing_loader = test_data.dataset_loader("test")

    model = SiameseModel(model_type=model_type, embedding_dim=embedding_dim).to(device)

    eval_criterion = nn.CosineEmbeddingLoss(margin=0.5)
    train_criterion = nn.CosineEmbeddingLoss(margin=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=Learning_rate_para["MAX_LR"], weight_decay=1e-2)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=end_epoch,
        eta_min=1e-6
    )
    print("Training using CosineAnnealingLR")
    best_acc = 0

    # if resume == True:
    #     begin_epoch = Loading_Checkpoint(path=checkpoint_path,
    #                                      model=model,
    #                                      optimizer=optimizer,
    #                                      scheduler=scheduler,
    #                                      device=device)
    #     best_acc = Get_Max_Acc(metrics_path)

    for epoch in range(begin_epoch, end_epoch):
        train_metrics = train(epoch, 
                                end_epoch,
                                model=model, 
                                loader=training_loader, 
                                criterion=train_criterion, 
                                optimizer=optimizer, 
                                device=device)
        train_loss = train_metrics.avg_cosemb_loss
        scheduler.step()
        print()
        train_data.regenerate_pair() #Tạo mới pairs
        val_metrics = validate(epoch, end_epoch, model, testing_loader, eval_criterion, device)
        val_loss = val_metrics.avg_cosemb_loss
        print()

        # if save_checkpoint == True:
        #     Saving_Checkpoint(epoch=epoch, 
        #                     model=model, 
        #                     optimizer=optimizer, 
        #                     scheduler=scheduler,
        #                     last_epoch=epoch, 
        #                     path=checkpoint_path)

        print("Epoch [{0}/{1}]: Training loss: {2}".
            format(epoch, end_epoch, train_loss))
        print("Epoch [{0}/{1}]: Validation loss: {2}".
            format(epoch, end_epoch, val_loss))
        # if val_acc > best_acc:
        #     if save_best == True:
        #         print("Validation accuracy increase from {0}% to {1}% at epoch {2}. Saving best result".
        #             format(round(best_acc * 100.0, 2), round(val_acc * 100.0, 2),  epoch))
        #         Saving_Best(model, best_path)
        #     else:
        #         print("Validation accuracy increase from {0}% to {1}% at epoch {2}".
        #             format(round(best_acc * 100.0, 2), round(val_acc * 100.0, 2),  epoch))
        #     best_acc = val_acc
        #     epochs_no_improve = 0  # reset patience
        # else:
        #     epochs_no_improve += 1
        # if save_metrics:
        #     Saving_Metric2(epoch=epoch, 
        #                    train_loss=train_loss,
        #                    train_acc=train_acc,
        #                    train_precision=train_metrics.precision_macro,
        #                    train_recall=train_metrics.recall_macro,
        #                    train_f1=train_metrics.f1_macro, 
        #                    val_loss=val_loss,
        #                    val_acc=val_acc,
        #                    val_precision=val_metrics.precision_macro,
        #                    val_recall=val_metrics.recall_macro,
        #                    val_f1=val_metrics.f1_macro, 
        #                    path=metrics_path)
        # if epochs_no_improve >= patience and early_stopping == True:
        #     print("Early stopping triggered at epoch {0}".format(epoch))
        #     break
        print()

    
if __name__ == '__main__':
    main()

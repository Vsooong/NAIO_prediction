# coding=utf-8
import torch
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import time
import torch.nn as nn
from Forecast_Data import naio_data
from Configs import cfg
from Forecast_Model import getEFModel
from utils import visual_AE

import os
from torch.utils.data import Dataset, DataLoader

# Reproducibility.
torch.manual_seed(cfg.GLOBAL.SEED)
if cfg.GLOBAL.CUDA:
    torch.cuda.manual_seed(cfg.GLOBAL.SEED)

evaluateL2 = nn.MSELoss(reduction='sum')
evaluateL1 = nn.L1Loss(reduction='sum')
use_reconstruct = True


# train on training dataset
def train(dataloader, model, criterion, optim, epoch):
    model.train()
    total_loss = 0
    n_samples = 0
    val_criter = nn.MSELoss(reduction='sum').to(cfg.GLOBAL.DEVICE)
    for minibatch in dataloader:
        X, Y = minibatch
        X = X.transpose(0, 1)
        X = X.to(device)
        Y = Y.to(device)
        optim.zero_grad()
        output = model(X)
        output = output.transpose(0, 1)
        loss = criterion(output, Y)
        loss.backward()
        total_loss += val_criter(output, Y).data.item()
        optim.step()
        n_samples += int(Y.size(0) * Y.size(1))

    total_loss = total_loss / n_samples / Y.size(-2) / Y.size(-1)
    time_used = time.time() - start_time
    print('\nEpoch: ', str(epoch))
    print('Time consumed: ', time_used)
    print('Prediction error: ', total_loss)
    return total_loss, time_used


if __name__ == '__main__':
    start_time = 0
    past_window = cfg.GLOBAL.IN_LEN
    forecast_step = cfg.GLOBAL.OUT_HORIZON
    base_root = cfg.GLOBAL.DATA_BASE_PATH

    train_dataset = naio_data(base_root)
    dataloader = DataLoader(train_dataset, batch_size=40, shuffle=True, num_workers=4)

    encoder_forecaster, optim, criterion = getEFModel()
    model = encoder_forecaster
    device = cfg.GLOBAL.DEVICE
    # if cfg.GLOBAL.CUDA:
    #     if torch.cuda.device_count() > 1:
    #         print("Let's use", torch.cuda.device_count(), "GPUs!")
    #         model = nn.DataParallel(model)
    model = model.to(device)

    best_val = 1000000
    print('-' * 89)
    print('training examples: %d' % train_dataset.target.size(0))
    nParams = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    print('number of parameters: %d' % nParams)

    #     # While training you can press Ctrl + C to stop it.
    try:
        print('Training start')
        start_time = time.time()
        loss1_list = []
        time_list = []
        for epoch in range(1, cfg.STMODEL.EPOCHS + 1):
            epoch_start_time = time.time()

            train_loss, time_used = train(dataloader, model, criterion, optim, epoch)
            loss1_list.append(train_loss)
            time_list.append(time_used)
            val_loss=train_loss
            if val_loss < best_val:
                print('\n Val loss improves to: \n', val_loss)
                with open(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'naio.pth'), 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_val = val_loss
        print(time_list)
        print(loss1_list)
    except KeyboardInterrupt:
        print('Exiting from training')

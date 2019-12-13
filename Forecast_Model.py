# coding=utf-8
import warnings
import os
warnings.filterwarnings("ignore")
import torch.nn as nn
from Configs import cfg
from models.forecaster import Forecaster
from models.encoder import Encoder
from models.model import EF
from torch.optim import lr_scheduler
import torch
from models.net_params import encoder_params, forecaster_params, convlstm_encoder_params, \
    convlstm_forecaster_params
from utils import makeOptimizer



def getEFModel():
    OPTIMIZER = 'adamw'
    LR = 0.002
    WEIGHT_DECAY = 0.3


    if cfg.STMODEL.NAME == 'TrajGRU':
        encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)
        forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(cfg.GLOBAL.DEVICE)
    else:
        encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(
            cfg.GLOBAL.DEVICE)
        forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(
            cfg.GLOBAL.DEVICE)

    encoder_forecaster = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)

    # load saved autoencoder model
    load_path=os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR,'AE-mvts.pth')
    if os.path.exists(load_path):
        encoder_forecaster.load_state_dict(torch.load(load_path,map_location=cfg.GLOBAL.DEVICE))

    optim = makeOptimizer(encoder_forecaster.parameters(), OPTIMIZER, LR, WEIGHT_DECAY)
    # loss and evaluate
    if cfg.GLOBAL.L1LOSS:
        criterion = nn.L1Loss(reduction='sum').to(cfg.GLOBAL.DEVICE)
    else:
        criterion = nn.MSELoss(reduction='sum').to(cfg.GLOBAL.DEVICE)

    # for param in encoder_forecaster.parameters():
    #     param.requires_grad=False

    return encoder_forecaster,optim,criterion

if __name__ == '__main__':
    input = torch.rand(20, 10, 5, 25, 53).to(cfg.GLOBAL.DEVICE)
    import time

    input = input.transpose(0, 1)
    model = getEFModel()[0]
    start = time.time()
    output = model(input)
    print(output.shape)
    end = time.time()
    print(end - start)

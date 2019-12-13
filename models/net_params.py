from Configs import cfg
from collections import OrderedDict
from models.trajGRU import TrajGRU
from models.convLSTM import ConvLSTM
import torch.nn as nn
from utils import Flatten, UnFlatten

batch_size = cfg.STMODEL.BATCH_SZIE

# build model
encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [5, 64, 3, 1, 1]}),
        # OrderedDict({'conv2_leaky_1': [20, 48, 3, 1, 1]}),
        # OrderedDict({'conv3_leaky_1': [48, 64, 3, 1, 1]}),
    ],

    [
        TrajGRU(input_channel=64, num_filter=64, b_h_w=(batch_size, 25, 53), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.STMODEL.RNN_ACT_TYPE),
        #
        # TrajGRU(input_channel=48, num_filter=48, b_h_w=(batch_size, 25, 53), zoneout=0.0, L=13,
        #         i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
        #         h2h_kernel=(5, 5), h2h_dilate=(1, 1),
        # #         act_type=cfg.STMODEL.RNN_ACT_TYPE),
        # TrajGRU(input_channel=64, num_filter=64, b_h_w=(batch_size, 25, 53), zoneout=0.0, L=9,
        #         i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
        #         h2h_kernel=(3, 3), h2h_dilate=(1, 1),
        #         act_type=cfg.STMODEL.RNN_ACT_TYPE)
    ]
]

forecaster_params = [
    [
        # OrderedDict({'deconv1_leaky_1': [64, 48, 3, 1, 1]}),
        # OrderedDict({'deconv2_leaky_1': [48, 20, 3, 1, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 5, 3, 1, 1],
            # 'deconv4_leaky_2': [8, 8, 5, 1, 0],
            'conv3_3': [5, 1, 1, 1, 0]
        }),
    ],

    [
        TrajGRU(input_channel=64, num_filter=64, b_h_w=(batch_size, 25, 53), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=cfg.STMODEL.RNN_ACT_TYPE),

        # TrajGRU(input_channel=48, num_filter=48, b_h_w=(batch_size, 25, 53), zoneout=0.0, L=13,
        #         i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
        #         h2h_kernel=(5, 5), h2h_dilate=(1, 1),
        #         act_type=cfg.STMODEL.RNN_ACT_TYPE),
        # TrajGRU(input_channel=20, num_filter=20, b_h_w=(batch_size, 25, 53), zoneout=0.0, L=9,
        #         i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
        #         h2h_kernel=(5, 5), h2h_dilate=(1, 1),
        #         act_type=cfg.STMODEL.RNN_ACT_TYPE)
    ]
]

conv2d_params = OrderedDict({
    'conv1_relu_1': [5, 64, 7, 5, 1],
    'conv2_relu_1': [64, 192, 5, 3, 1],
    'conv3_relu_1': [192, 192, 3, 2, 1],
    'deconv1_relu_1': [192, 192, 4, 2, 1],
    'deconv2_relu_1': [192, 64, 5, 3, 1],
    'deconv3_relu_1': [64, 64, 7, 5, 1],
    'conv3_relu_2': [64, 20, 3, 1, 1],
    'conv3_3': [20, 20, 1, 1, 0]
})


# build model
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [5, 5, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [5, 5, 3, 1, 1]}),
        OrderedDict({'conv3_leaky_1': [5, 5, 3, 1, 1]}),
    ],

    [
        ConvLSTM(input_channel=5, num_filter=5, b_h_w=(batch_size, 25, 53),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=5, num_filter=5, b_h_w=(batch_size, 25, 53),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=5, num_filter=5, b_h_w=(batch_size, 25, 53),
                 kernel_size=3, stride=1, padding=1),
    ]
]

convlstm_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [5, 5, 3, 1, 1]}),
        OrderedDict({'deconv2_leaky_1': [5, 5, 3, 1, 1]}),
        OrderedDict({
            # 'deconv3_leaky_1': [10, 8, 7, 5, 0],
            # 'deconv4_leaky_2': [8, 8, 5, 1, 0],
            'conv3_3': [5, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM(input_channel=5, num_filter=5, b_h_w=(batch_size, 25, 53),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=5, num_filter=5, b_h_w=(batch_size, 25, 53),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=5, num_filter=5, b_h_w=(batch_size, 25, 53),
                 kernel_size=3, stride=1, padding=1),
    ]
]

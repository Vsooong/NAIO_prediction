import numpy as np
import os
import torch
import torch.nn.functional as F
from collections import OrderedDict


class edict(OrderedDict):
    """Using OrderedDict for the `easydict` package
    See Also https://pypi.python.org/pypi/easydict/
    """

    def __init__(self, d=None, **kwargs):
        super(edict, self).__init__()
        if d is None:
            d = OrderedDict()
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        # special handling of self.__root and self.__map
        if name.startswith('_') and (name.endswith('__root') or name.endswith('__map')):
            super(edict, self).__setattr__(name, value)
        else:
            if isinstance(value, (list, tuple)):
                value = [self.__class__(x)
                         if isinstance(x, dict) else x for x in value]
            else:
                value = self.__class__(value) if isinstance(value, dict) else value
            super(edict, self).__setattr__(name, value)
            super(edict, self).__setitem__(name, value)

    __setitem__ = __setattr__


__C = edict()
__C.GLOBAL = edict()
__C.GLOBAL.DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
__C.GLOBAL.CUDA = True if torch.cuda.is_available() else False
__C.GLOBAL.SEED = 1111
for dirs in ['F:/Python_Project/TC_intensity_prediction/save/',
             '/home/dl/Desktop/NAIO_prediction/Save']:
    if os.path.exists(dirs):
        __C.GLOBAL.MODEL_SAVE_DIR = dirs
assert __C.GLOBAL.MODEL_SAVE_DIR is not None

for dirs in ['F:/data/sea data/',
             '/home/dl/data/sea_data/']:
    if os.path.exists(dirs):
        __C.GLOBAL.SEA_DATA = dirs

__C.INFARED = edict()
for dirs in ['/home/dl/data/lj_nao_data', 'D:/Python_Project/NAIO_prediction/data_lj_nao/']:
    if os.path.exists(dirs):
        __C.GLOBAL.DATA_BASE_PATH = dirs
__C.INFARED.IMG_SIZE = 256
__C.GLOBAL.IN_LEN = 3
__C.GLOBAL.OUT_HORIZON =[]
__C.GLOBAL.OUT_LEN = 10
__C.GLOBAL.STRIDE = 1  # The stride
__C.GLOBAL.TARGET_DIM = 1
__C.GLOBAL.INPUT_DIM=12
__C.GLOBAL.L1LOSS = False
__C.STMODEL = edict()
__C.STMODEL.NAME = 'TrajGRU'
# __C.STMODEL.NAME = 'Convlstm'
__C.STMODEL.CLIP = 1
__C.STMODEL.EPOCHS = 300
__C.STMODEL.BATCH_SZIE = 16

class activation():

    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError


__C.STMODEL.RNN_ACT_TYPE = activation('leaky', negative_slope=0.2, inplace=True)
cfg = __C

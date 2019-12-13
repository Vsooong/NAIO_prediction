from torch import nn
from collections import OrderedDict
import torch.optim as optim


def makeOptimizer(params, optimn, lr, weight_decay):
    optimn = str(optimn).lower()
    if optimn == 'sgd':
        optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.8)
    elif optimn == 'adagrad':
        optimizer = optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    elif optimn == 'adadelta':
        optimizer = optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    elif optimn == 'adamw':
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimn == 'adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

    else:
        raise RuntimeError("Invalid optim method")
    return optimizer

def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                 padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))


            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))


def visual_AE(images, title='', horizon=3, path=None):
    import matplotlib.pyplot as plt
    assert len(images) == horizon
    fig = plt.figure(figsize=(18, 6))
    columns = horizon
    rows = 1
    for i in range(1, columns * rows + 1):
        img = images[i - 1].squeeze().detach().numpy().astype(int)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    fig.suptitle(title, fontsize=16)
    if path is not None:
        plt.savefig(path)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.unsqueeze(-1).unsqueeze(-1)

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from module_creation_constant import *


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class Darknet(nn.Module):
    def __init__(self, config_file):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(config_file)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, input, CUDA):
        # Start from index 1, as 0 contains the 'net' block
        modules = self.blocks[1:]
        outputs = {}

        write = 0
        for index, module in enumerate(modules):
            module_type = module[TYPE]
            if module_type == CONVOLUTIONAL or module_type == UPSAMPLE:
                input = self.module_list[index](input)
            elif module_type == ROUTE:
                layers = module[LAYERS]
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - index

                if len(layers) == 1:
                    input = outputs[index + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - index

                    map1 = outputs[index + layers[0]]
                    map2 = outputs[index + layers[1]]

                    # Concatenate along depth.
                    input = torch.cat((map1, map2), 1)

            elif module_type == SHORTCUT:
                from_ = int(module["from"])
                input = outputs[index-1] + outputs[index+from_]





def parse_cfg(config_file):
    """
    Takes a configuration file

    Returns a list of blocks. Each block describes a block in the neural network to be built.
    A block is represented as a dictionary in the list.

    """

    with open(config_file, 'r') as file:
        lines = file.read().split('\n')
        lines = [line for line in lines if len(line) > 0]
        lines = [line for line in lines if line[0] != '#']
        lines = [line.strip() for line in lines]

    block = {}
    list_of_blocks = []

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                list_of_blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.strip()] = value.strip()
    list_of_blocks.append(block)

    return list_of_blocks


def create_modules(blocks):
    # Get info about the network - input and pre-processing
    net_info = blocks[0]
    module_list = nn.ModuleList()

    # Keep track of the number of filters in the layer that the convolutional layer is applied to. 3 (RGB channels)
    previous_filters = 3
    output_filters = []

    for index, block in enumerate(blocks[1:]):
        module = nn.Sequential()

        # Check the type of the block
        # Create a new module for that block
        # Append the module to module_list

        if block[TYPE] == CONVOLUTIONAL:
            # Get information about the layer
            activation = block[ACTIVATION]
            try:
                batch_normalise = int(block[BATCH_NORMALISE])
                bias = False
            except KeyError:
                batch_normalise = 0
                bias = True

            filters = int(block[FILTERS])
            padding = int(block[PAD])
            kernel_size = int(block[SIZE])
            stride = int(block[STRIDE])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer
            convolutional_layer = nn.Conv2d(previous_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), convolutional_layer)

            # Add the Batch Normaliser layer
            if batch_normalise:
                batch_norm_layer = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), batch_norm_layer)

            # Check the activation
            # It is either Linear or Leaky ReLU for YOLO
            if activation == LEAKY:
                activation_layer = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activation_layer)

        # If it's an upsampling layer we use Bilinear2dUpsampling
        elif block[TYPE] == UPSAMPLE:
            stride = int(block[STRIDE])
            upsample_player = nn.Upsample(scale_factor=2, mode=BILINEAR)
            module.add_module("upsample_{0}".format(index), upsample_player)

        elif block[TYPE] == ROUTE:
            block[LAYERS] = block[LAYERS].split(',')

            # Start of a route
            start = int(block[LAYERS][0])

            # End - if one exists
            try:
                end = int(block[LAYERS][1])
            except:
                end = 0

            # Positive annotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # Shortcut corresponds to skip connections
        elif block[TYPE] == SHORTCUT:
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut)

        # The YOLO (detection) layer
        elif block[TYPE] == YOLO:
            mask = block[MASK].split(",")
            mask = [int(value) for value in mask]

            anchors = block[ANCHORS].split(",")
            anchors = [int(anchor) for anchor in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection_layer = DetectionLayer(anchors)
            module.add_module("detection_{0}".format(index), detection_layer)

        module_list.append(module)
        previous_filters = filters
        output_filters.append(filters)

    return net_info, module_list


blocks = parse_cfg("cfg/yolov3.cfg")
print(create_modules(blocks))

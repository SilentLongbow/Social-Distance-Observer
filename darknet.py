from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.autograd import Variable
import numpy as np
from util import predict_transform

from module_creation_constant import *


def get_test_input():

    img = cv2.imread("dog-cycle-car.png")

    # Resize input dimension
    img = cv2.resize(img, (416,416))

    # BGR -> RGB | H X W C -> C X H X W
    img_ = img[:,:,::-1].transpose((2,0,1))

    # Add a channel at 0 (for the batch) | Normalise
    img_ = img_[np.newaxis,:,:,:]/255.0

    # Convert to float
    img_ = torch.from_numpy(img_).float()

    # Convert to Variable type
    img_ = Variable(img_)
    return img_


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
        self.header = None
        self.seen = None

    def load_weights(self, weight_file):
        # Open the file
        with open(weight_file) as file:
            """
            First 5 values are header information
            1. Major revision number
            2. Minor revision number
            3. Subversion number
            4, 5. Images seen by the network (during training
            """
            header = np.fromfile(file, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]

            weights = np.fromfile(file, dtype=np.float32)

            # Keeps track of where we are in the weights array
            pointer = 0
            for index in range(len(self.module_list)):
                module_type = self.block[index+1][TYPE]

                # If module_type is convolutional, load the weights
                # Otherwise, ignore.
                if module_type == CONVOLUTIONAL:
                    model = self.module_list[index]
                    try:
                        batch_normalise = int(self.blocks[index+1][BATCH_NORMALISE])
                    except Exception as exception:
                        batch_normalise = 0

                    convolution = model[0]

                    # Batch normalisations - Method of normalising the data to a range of [0, 1]
                    if batch_normalise:
                        """ Load the weights """
                        batch_norm_layer = model[1]
                        
                        # Get the number of weights from the Batch Norm layer
                        num_batch_norm_biases = batch_norm_layer.bias.numel()
                        
                        # Load the weights
                        batch_norm_biases = torch.from_numpy(weights[pointer:pointer + num_batch_norm_biases])
                        pointer += num_batch_norm_biases

                        batch_norm_weights = torch.from_numpy(weights[pointer: pointer + num_batch_norm_biases])
                        pointer += num_batch_norm_biases

                        batch_norm_running_mean = torch.from_numpy(weights[pointer: pointer + num_batch_norm_biases])
                        pointer += num_batch_norm_biases

                        batch_norm_running_var = torch.from_numpy(weights[pointer: pointer + num_batch_norm_biases])
                        pointer += num_batch_norm_biases

                        # Cast loaded weights into dimensions of model weights.
                        batch_norm_biases = batch_norm_biases.view_as(batch_norm_layer.bias.data)
                        batch_norm_weights = batch_norm_weights.view_as(batch_norm_layer.weight.data)
                        batch_norm_running_mean = batch_norm_running_mean.view_as(batch_norm_layer.running_mean)
                        batch_norm_running_var = batch_norm_running_var.view_as(batch_norm_layer.running_var)

                        # Copy the data to the model
                        batch_norm_layer.bias.data.copy_(batch_norm_biases)
                        batch_norm_layer.weight.data.copy_(batch_norm_weights)
                        batch_norm_layer.running_mean.data.copy_(batch_norm_running_mean)
                        batch_norm_layer.running_var.data.copy_(batch_norm_running_var)
                    else:
                        # Load the biases of the convolutional layer

                        # Number of biases
                        num_biases = convolution.bias.numel()

                        # Load the weights
                        conv_biases = torch.from_numpy(weights[pointer: pointer + num_biases])
                        pointer += num_biases

                        # Reshape the loaded weights according to the dimensions of the model weights
                        conv_biases = conv_biases.view_as(convolution.bias.data)

                        # And now copy the data back
                        convolution.bias.data.copy_(conv_biases)

                    # Now load the weights for the Convolutional layers!
                    num_weights = convolution.weight.numel()

                    # Do the same as above for the weights
                    conv_weights = torch.from_numpy(weights[pointer: pointer + num_weights])
                    pointer += num_weights

                    conv_weights = conv_weights.view_as(convolution.weight.data)
                    convolution.weight.data.copy_(conv_weights)

    def forward(self, layer_input, CUDA):
        print("Forward called")
        # Start from index 1, as 0 contains the 'net' block
        modules = self.blocks[1:]
        outputs = {}

        # Indicates whether the first detection tensor is available or not.
        write = 0
        for index, module in enumerate(modules):
            module_type = module[TYPE]
            if module_type == CONVOLUTIONAL or module_type == UPSAMPLE:
                layer_input = self.module_list[index](layer_input)
            elif module_type == ROUTE:
                layers = module[LAYERS]
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - index

                if len(layers) == 1:
                    layer_input = outputs[index + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - index

                    map1 = outputs[index + layers[0]]
                    map2 = outputs[index + layers[1]]

                    # Concatenate along depth.
                    layer_input = torch.cat((map1, map2), 1)

            elif module_type == SHORTCUT:
                from_ = int(module["from"])
                layer_input = outputs[index-1] + outputs[index+from_]

            elif module_type == YOLO:

                anchors = self.module_list[index][0].anchors
                # Get the input dimensions
                input_dimensions = int(self.net_info[HEIGHT])

                # Get the number of classes
                num_classes = int(module[CLASSES])

                # Transform
                layer_input = layer_input.data
                layer_input = predict_transform(layer_input, input_dimensions, anchors, num_classes, CUDA)
                if not write:
                    detections = layer_input
                    write = 1
                else:
                    detections = torch.cat((detections, layer_input), 1)
            outputs[index] = layer_input
        return detections


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


model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")

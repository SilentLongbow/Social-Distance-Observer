from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def predict_transform(prediction, input_dimensions, anchors, num_classes, CUDA=True):
    """ Takes a detected feature map and turns it into a 2D tensor. """

    batch_size = prediction.size(0)
    stride = input_dimensions // prediction.size(2)
    grid_size = input_dimensions // stride
    bounding_box_attributes = 5 + num_classes
    num_anchors = len(anchors)

    if CUDA:
        prediction = prediction.cuda()

    prediction = prediction.view(batch_size, bounding_box_attributes*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bounding_box_attributes)

    anchors = [(anchor[0] / stride, anchor[1] / stride) for anchor in anchors]

    # Transform output according to sigmoid function

    # Sigmoid the centre_X, centre_Y and object confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add grid offsets to the centre coordinates prediction
    grid = np.arange(grid_size)
    x_coordinates, y_coordinates = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(x_coordinates).view(-1, 1)
    y_offset = torch.FloatTensor(y_coordinates).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    # Apply anchors to dimensions of bounding box
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    # Apply sigmoid activation to class scores
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # Resize detections map to size of input image
    prediction[:,:,:4] *= stride

    return prediction


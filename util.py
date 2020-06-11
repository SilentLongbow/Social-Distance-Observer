from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

"""
NOTE: .view() reshapes a tensor to match the dimensions given.
If you are unsure about the number of rows you want, but do know about how many columns, you can use -1.
    - Or vice versa.
    - Only one parameter can be '-1'
"""


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

    prediction[:, :, :2] += x_y_offset

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


def write_results(prediction, confidence, num_classes, non_max_supp_conf=0.4):
    """
    Takes in the predicion, confidence (objectness score threshold), number of classes and the Non-Maximal suppression
    intercept over union.
    """

    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    box_corner = prediction.new(prediction.shape)

    # Convert centre x to top-left corner x
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]) / 2
    # Convert centre y to top-left corner y
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]) / 2
    # Convert height to bottom-right corner x
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]) / 2
    # Convert width to bottom-right corner y
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]) / 2

    prediction[:, :, :4] = box_corner[:, :, :4]

    # Starting NMS pre-processing
    batch_size = prediction.size(0)

    write = False

    for index in range (batch_size):
        image_prediction = prediction[index]
        # Start confidence thresholding

        max_confidence, max_confidence_score = torch.max(image_prediction[:, 5:5+num_classes], 1)
        max_confidence = max_confidence.float().unsqueeze(1)
        max_confidence_score = max_confidence_score.float().unsqueeze(1)
        sequence = image_prediction[:, :5], max_confidence, max_confidence_score
        image_prediction = torch.cat(sequence, 1)

        # Returns an array of indices of all non-zero values within the tensor.
        non_zero_index = torch.nonzero(image_prediction[:, 4])
        try:
            image_prediction_ = image_prediction[non_zero_index.squeeze(), :].view(-1, 7)
        except:
            # No detections if we get here
            continue

        # Above code will not raise an exception for no detection
        # So perform this action
        if image_prediction_.shape[0] == 0:
            continue

        # Get the different classes that the network has detected in the image
        # The -1 index holds the class index
        img_classes = unique(image_prediction_[:, -1])

        for object_class in img_classes:
            class_mask = image_prediction_ * (image_prediction_[:, 1] == object_class).float().unsqueeze(1)
            class_mask_index = torch.nonzero(class_mask[:, -1]).squeeze()
            image_prediction_class = image_prediction_[class_mask_index].view(-1, 7)

            # Sort the detections such that the entry with the maximum objectness confidence is at the top
            confidence_sort_index = torch.sort(image_prediction_class[:, 4], descending=True)[1]
            image_prediction_class = image_prediction_class[confidence_sort_index]

            # Get the number of detections
            idx = image_prediction_class.size(0)

            for i in range(idx):
                # Get the IoUs for all the boxes that come after the one we're looking at in the loop.
                try:
                    # bbox_iou(bounding_box_row, tensor_of_multiple_rows_of_bounding_boxes) -> tensor_of_ious_of_bounding_box
                    ious = bbox_iou(image_prediction_class[i].unsqueeze(0), image_prediction_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # Zero out all the detections that have IoU > threshold
                iou_mask = (ious < non_max_supp_conf).float().unsqueeze(1)
                image_prediction_class[i+1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_index = torch.nonzero(image_prediction_class[:, 4]).squeeze()
                image_prediction_class = image_prediction_class[non_zero_index].view(-1, 7)

def unique(tensor):
    """
    Get the unique classes detected within an image.
    """
    tensor_np = tensor.cpu().numpy()
    # Return sorted unique elements of array
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_resolution = tensor.new(unique_tensor.shape)
    tensor_resolution.copy_(unique_tensor)
    return tensor_resolution

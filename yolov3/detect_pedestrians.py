import os
import argparse
from datetime import datetime

import cv2
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

from .models import *
from .utils.utils import load_classes, non_max_suppression


class PedestrianDetector:

    def __init__(self, model_definition, weights_path, image_size, class_definitions_path, device):
        self.detector_model = Darknet(model_definition, image_size).to(device)
        if weights_path.endswith(".weights"):
            self.detector_model.load_darknet_weights(weights_path)
        else:
            raise ValueError("Weights file is not supported: {}\n Use a '.weights' file.".format(weights_path))
        self.detector_model.eval()
        self.classes = load_classes(class_definitions_path)
        self.detections = None

    def perform_pedestrian_detection(self, input_image, confidence_threshold, iou_threshold):
        with torch.no_grad():
            # Get all bounding box detections
            detections = self.detector_model(input_image)

            # Cull any bounding boxes that don't meet confidence threshold and perform nms on the others
            # Returns list containing a detections tensor. We just want that tensor.
            self.detections = non_max_suppression(detections, confidence_threshold, iou_threshold)[0]

    def restore_bounding_boxes(self, current_dim, original_shape):
        """ Rescales bounding boxes to the original shape """
        orig_h, orig_w = original_shape
        # The amount of padding that was added
        pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
        pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
        # Image height and width after padding is removed
        unpad_h = current_dim - pad_y
        unpad_w = current_dim - pad_x
        # Rescale bounding boxes to dimension of original image
        self.detections[:, 0] = ((self.detections[:, 0] - pad_x // 2) / unpad_w) * orig_w
        self.detections[:, 1] = ((self.detections[:, 1] - pad_y // 2) / unpad_h) * orig_h
        self.detections[:, 2] = ((self.detections[:, 2] - pad_x // 2) / unpad_w) * orig_w
        self.detections[:, 3] = ((self.detections[:, 3] - pad_y // 2) / unpad_h) * orig_h
        return self.detections

    def cull_non_pedestrian_detections(self):
        class_pred_index = 6
        pedestrian_detections = []
        for detection in self.detections:
            class_name_index = detection[class_pred_index]
            if self.classes[int(class_name_index)] == "person":
                pedestrian_detections.append(detection)
        if len(pedestrian_detections) > 0:
            self.detections = torch.stack(pedestrian_detections)
        else:
            self.detections = None


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def opencv_image_to_tensor(image, image_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image)
    image, _ = pad_to_square(image, pad_value=0)
    image = resize(image, image_size)
    image = image.unsqueeze(0)
    return image


def get_capture(use_webcam, video_input):
    if use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_input)
    return cap


def set_output(input_capture):
    filename = "video_output/detections-{}".format(datetime.now())
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    resolution = int(input_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(input_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = input_capture.get(cv2.CAP_PROP_FPS)
    return cv2.VideoWriter(filename, fourcc, frame_rate, resolution)


def convert_image_for_detection(frame, yolo_input_size, tensor_type):
    image = opencv_image_to_tensor(frame, yolo_input_size)
    return Variable(image.type(tensor_type))
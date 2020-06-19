import os
import argparse
from datetime import datetime

import cv2
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

from models import *
from utils.utils import load_classes, non_max_suppression, rescale_boxes


class PedestrianDetector:

    def __init__(self, model_definition, weights_path, image_size, device, class_definitions_path, ):
        self.detector_model = Darknet(model_definition, image_size).to(device)
        if weights_path.endswith(".weights"):
            self.detector_model.load_darknet_weights(weights_path)
        else:
            raise ValueError("Weights file is not supported: {}\n Use a '.weights' file.".format(weights_path))
        self.detector_model.eval()
        self.classes = load_classes(class_definitions_path)

    def perform_pedestrian_detection(self, input_image, confidence_threshold, iou_threshold):
        with torch.no_grad():
            # Get all bounding box detections
            detections = self.detector_model(input_image)

            # Cull any bounding boxes that don't meet confidence threshold and perform nms on the others
            # Returns list containing a detections tensor. We just want that tensor.
            return non_max_suppression(detections, confidence_threshold, iou_threshold)[0]


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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="path to the input image")
    parser.add_argument("--model-definition", type=str, default="config/yolov3.cfg",
                        help="path to the pre-trained model")
    parser.add_argument("--weights-path", type=str, default="weights/yolov3.weights",
                        help="path to pre-trained weights file")
    parser.add_argument("--class-path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf-threshold", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--non-max-supp-threshold", type=float, default=0.4,
                        help="iou threshold for non-maximum suppression")
    parser.add_argument("--image-size", type=int, default=416, help="the size of each image dimension")
    parser.add_argument("--webcam", type=bool, default=False, help="whether the device's webcam should be used")
    return parser.parse_args()


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


def prepare_image_input(frame, options, Tensor):
    image = opencv_image_to_tensor(frame, options.image_size)
    return Variable(image.type(Tensor))


def initialise_detector(model_definition, weights_path, image_size, device, class_path):
    return PedestrianDetector(options.model_definition, options.weights_path,
                                             options.image_size, device, options.class_path)




if __name__ == '__main__':
    options = parse_arguments()
    os.makedirs("video_output", exist_ok=True)

    while capture.isOpened():
        successfully_retrieved, frame = capture.read()

        if not successfully_retrieved:
            break

        yolo_input_image = prepare_image_input(frame, options)
        detections = pedestrian_detector\
            .perform_pedestrian_detection(yolo_input_image, options.conf_threshold, options.non_max_supp_threshold)

        if detections is not None:
            detections = rescale_boxes(detections, options.image_size, frame.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            for x1, y1, x2, y2, object_confidence, class_confidence, class_prediction_index in detections:
                if int(class_prediction_index) == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    text = "{0} - {1:.3f}%".format(pedestrian_detector.classes[int(class_prediction_index)], object_confidence)
                    cv2.putText(frame, text, (x1, y1), 0, 0.5, (0, 255, 0), 2)

        output.write(frame)
        cv2.imshow("Detections", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

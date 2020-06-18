import os
import argparse

import cv2
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

from models import *
from utils.utils import load_classes, non_max_suppression, rescale_boxes


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


def initialise_model(model_definition, image_size, device, weights_path):
    darknet_model = Darknet(model_definition, img_size=image_size).to(device)
    if weights_path.endswith(".weights"):
        # Load Darknet weights
        darknet_model.load_darknet_weights(weights_path)
    else:
        raise Exception("No weight file provided")
    darknet_model.eval()
    return darknet_model


def opencv_image_to_tensor(image, image_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image)
    image, _ = pad_to_square(image, pad_value=0)
    image = resize(image, image_size)
    image = image.unsqueeze(0)
    return image


# def display_loop(capture):
#     while capture.isOpened():
#         ret, frame = cap.read()
#
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # TODO - Get image (maybe convert to RGB), perform transform to tensor, pad to sqaure, resize to img_size
#
#         if not ret:
#             break
#
#         new_image = cv2.resize(frame, (1280, 720))
#         cv2.imshow("Display window", new_image)
#
#         if cv2.waitKey(17) == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="path to the input image")
    parser.add_argument("--model-definition", type=str, default="config/yolov3.cfg", help="path to the pre-trained model")
    parser.add_argument("--weights-path", type=str, default="weights/yolov3.weights", help="path to pre-trained weights file")
    parser.add_argument("--class-path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf-threshold", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--non-max-supp-thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--image-size", type=int, default=416, help="the size of each image dimension")
    parser.add_argument("--webcam", type=bool, default=False, help="whether the device's webcam should be used")

    options = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("video_output", exist_ok=True)

    # Establish YOLO model.
    detector_model = initialise_model(options.model_definition, options.image_size, device, options.weights_path)

    classes = load_classes(options.class_path)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    if options.webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(options.input_path)

    # display_loop(cap)

    while cap.isOpened():
        successfully_retrieved, frame = cap.read()

        if not successfully_retrieved:
            break

        yolo_input_image = opencv_image_to_tensor(frame, options.image_size)
        yolo_input_image = Variable(yolo_input_image.type(Tensor))

        with torch.no_grad():
            # Gets all detections found by the model
            detections = detector_model(yolo_input_image)

            """ Performs non-max suppression to only get those detections with highest confidence.
            Returns a list containing detection tensors. Just actually want the tensor itself.
            """
            detections = non_max_suppression(detections, options.conf_threshold, options.non_max_supp_thres)[0]

        if detections is not None:
            detections = rescale_boxes(detections, options.image_size, frame.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            for x1, y1, x2, y2, object_confidence, class_confidence, class_prediction_index in detections:
                if int(class_prediction_index) == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

        cv2.imshow("Detections", frame)


        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

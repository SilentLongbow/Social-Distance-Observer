import argparse
from datetime import datetime

import cv2
import torch

import yolov3.detect_pedestrians as detector

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="path to the input image")
    parser.add_argument("--model-definition", type=str, default="yolov3/config/yolov3.cfg",
                        help="path to the pre-trained model")
    parser.add_argument("--weights-path", type=str, default="yolov3/weights/yolov3.weights",
                        help="path to pre-trained weights file")
    parser.add_argument("--class-path", type=str, default="yolov3/data/coco.names", help="path to class label file")
    parser.add_argument("--conf-threshold", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--non-max-supp-threshold", type=float, default=0.4,
                        help="iou threshold for non-maximum suppression")
    parser.add_argument("--image-size", type=int, default=416, help="the size of each image dimension")
    return parser.parse_args()


def get_output(capture):
    filename = "video_output/detections-{}".format(datetime.now())
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    resolution = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cv2.VideoWriter(filename, fourcc, frame_rate, resolution)


def processing_loop(pedestrian_detector, args, capture, output):
    confidence_threshold = args.conf_threshold
    non_max_suppression_threshold = args.non_max_supp_threshold
    yolo_image_size = args.image_size
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    while capture.isOpened():
        has_frame, frame = capture.read()

        if not has_frame:
            break

        detection_input_image = detector.prepare_image_input(frame, args, Tensor)
        detections = pedestrian_detector.perform_pedestrian_detection(detection_input_image,
                                                                      confidence_threshold,
                                                                      non_max_suppression_threshold)
        # TODO Need to get bounding boxes as bb_left, bb_top, width, height as per MOT instructions
        if detections is not None:
            for detection in detections:
                print()


if __name__ == '__main__':
    args = parse_command_line_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pedestrian_detector = detector.PedestrianDetector(args.model_definition,
                                                      args.weights_path,
                                                      args.image_size,
                                                      device,
                                                      args.class_path)

    capture = cv2.VideoCapture(args.input_path)
    output_file = get_output(capture)
    processing_loop(pedestrian_detector, args, capture, output_file)




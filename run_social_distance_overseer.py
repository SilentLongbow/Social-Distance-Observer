import argparse
import os
from datetime import datetime

import cv2
import torch
import numpy as np

from deepsort import *
import yolov3.detect_pedestrians as detector
from yolov3.utils import utils

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
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    resolution = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cv2.VideoWriter(filename, fourcc, frame_rate, resolution)


def reformat_detections(detections):
    detection_list = []
    confidence_list = []
    for x1, y1, x2, y2, object_conf, class_conf, class_index in detections:
        x1, x2, y1, y2 = float(x1), float(x2), float(y1), float(y2)
        object_conf = float(object_conf)
        width = x2 - x1
        height = y2 - y1
        detection_list.append([x1, y1, width, height])
        confidence_list.append(object_conf)
    detection_list = np.array(detection_list)
    confidence_list = np.array(confidence_list)
    return detection_list, confidence_list


def processing_loop(pedestrian_detector, pedestrian_tracker,args, capture, output):
    confidence_threshold = args.conf_threshold
    non_max_suppression_threshold = args.non_max_supp_threshold
    yolo_image_size = args.image_size
    ids_in_violaton = set()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    while capture.isOpened():
        has_frame, frame = capture.read()

        if not has_frame:
            break

        detection_input_image = detector.prepare_image_input(frame, args, Tensor)
        pedestrian_detector.perform_pedestrian_detection(detection_input_image,
                                                                      confidence_threshold,
                                                                      non_max_suppression_threshold)
        if pedestrian_detector.detections is not None:
            pedestrian_detector.detections = utils.rescale_boxes(pedestrian_detector.detections,
                                                                 args.image_size, frame.shape[:2])
            pedestrian_detector.cull_non_pedestrian_detections()
            if pedestrian_detector.detections is not None:
                xywh_detections, confidences = reformat_detections(pedestrian_detector.detections)
                tracker, detections_class = pedestrian_tracker.run_deep_sort(frame, confidences, xywh_detections)
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    id_num = str(track.track_id)
                    bbox = track.to_tlbr()
                    paint_red = False
                    if len(tracker.tracks) >= 2:
                        for other_track in tracker.tracks:
                            if track != other_track:
                                track_centroid = track.to_x0y0()
                                other_track_centroid = other_track.to_x0y0()
                                distance = np.linalg.norm(track_centroid - other_track_centroid)
                                if distance <= 50:
                                    cv2.line(frame, (int(track_centroid[0]), int(track_centroid[1])),
                                             (int(other_track_centroid[0]), int(other_track_centroid[1])),
                                             (0, 255, 0), 1)
                                    paint_red = True
                                    ids_in_violaton.add(id_num)
                        # # Draw bbox from tracker.
                    colour = (0, 0, 255) if paint_red else (255, 255, 255)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colour, 1)
                    cv2.putText(frame, "{}".format(id_num), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 100, (0, 255, 0), 1)
        cv2.putText(frame, "Total at risk: {}".format(len(ids_in_violaton)), (0, frame.shape[0] - 10), 0, 1, (0,0,255), 4)
        cv2.imshow('frame', frame)
        output_file.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == '__main__':
    args = parse_command_line_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pedestrian_detector = detector.PedestrianDetector(args.model_definition,
                                                      args.weights_path,
                                                      args.image_size,
                                                      device,
                                                      args.class_path)

    # Initialise tracker
    pedestrian_tracker = deepsort_rbc()

    # capture = cv2.VideoCapture(args.input_path)
    capture = cv2.VideoCapture(0)

    output_file = get_output(capture)
    processing_loop(pedestrian_detector, pedestrian_tracker, args, capture, output_file)




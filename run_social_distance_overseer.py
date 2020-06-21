import argparse
import os
import sys
from datetime import datetime

import cv2
import torch
import numpy as np

from deep_sort_python import deepsort
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
    parser.add_argument("--min-distance", type=int, default=50, help="social distance pixel threshold")
    parser.add_argument("--track-detections", type=bool, default=True,
                        help="perform tracking on top of YOLOv3 detections")
    return parser.parse_args()


def get_output(capture):
    filename = "video_output/detections-{}".format(datetime.now())
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    resolution = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cv2.VideoWriter(filename, fourcc, frame_rate, resolution)


def reformat_detections(detections):
    """ Converts all detections tensors into two ndarrays of [x1, y1, w, h], [obj_confidence] values. """
    cloned_detections = detections.clone()
    np_detections = cloned_detections[:, :4]
    np_detections[:, 2] = np_detections[:, 2] - np_detections[:, 0]
    np_detections[:, 3] = np_detections[:, 3] - np_detections[:, 1]
    np_detections = np_detections.numpy()
    np_confidences = cloned_detections[:, 4].numpy()
    return np_detections, np_confidences


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

        detection_input_image = detector.convert_image_for_detection(frame, args, Tensor)
        pedestrian_detector.perform_pedestrian_detection(detection_input_image,
                                                                      confidence_threshold,
                                                                      non_max_suppression_threshold)
        if pedestrian_detector.detections is not None:
            pedestrian_detector.detections = utils.rescale_boxes(pedestrian_detector.detections,
                                                                 args.image_size, frame.shape[:2])
            pedestrian_detector.cull_non_pedestrian_detections()
            if pedestrian_detector.detections is not None:
                xywh_detections, confidences = reformat_detections(pedestrian_detector.detections)
                tracker, detections = pedestrian_tracker.run_deep_sort(frame, confidences, xywh_detections)
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
                    cv2.rectangle(img=frame, pt1=(int(bbox[0]), int(bbox[1])), pt2=(int(bbox[2]), int(bbox[3])), color=colour, thickness=1)
                    cv2.putText(frame, "{}".format(id_num), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 100, (0, 255, 0), 1)
        cv2.putText(frame, "Total at risk: {}".format(len(ids_in_violaton)), (0, frame.shape[0] - 10), 0, 1, (0,0,255), 4)
        cv2.imshow('frame', frame)
        output.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


class PedestrianObserver:

    def __init__(self, args, capture):
        self.args = args
        self.capture = capture
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = detector.PedestrianDetector(self.args.model_definition,
                                                    self.args.weights_path,
                                                    self.args.image_size,
                                                    self.args.class_path,
                                                    self.device)

        # Add the deep_sort_python directory to sys as the pre-trained model needs deep_sort_python as working directory
        sys.path.append("deep_sort_python")
        self.pedestrian_tracker = deepsort.deepsort_rbc()
        sys.path.remove("deep_sort_python")

        self.pedestrians_within_distance_threshold = set()
        self.distance_threshold = self.args.min_distance
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.confidence_threshold = self.args.conf_threshold
        self.iou_threshold = self.args.non_max_supp_threshold

    def social_distance_loop(self):
        """
        Reads each frame in the opened capture, firing off calls to the object detector first.
        One the detector has completed, it launches the object tracker.
        Once the objects have been tracked, the distance calculations are performed.
        """
        while self.capture.isOpened():
            retrieved, frame = capture.read()
            if not retrieved:
                break
            self.perform_object_detection(frame)
            if self.args.track_detections:
                self.deep_sort_loop(frame)
            display_image("Social Distancing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def perform_object_detection(self, frame):
        detection_image = detector.convert_image_for_detection(frame, self.args.image_size, self.Tensor)
        self.detector.perform_pedestrian_detection(detection_image, self.confidence_threshold, self.iou_threshold)
        if self.detector.detections is not None:
            self.detector.restore_bounding_boxes(self.args.image_size, frame.shape[:2])
            self.detector.cull_non_pedestrian_detections()

    def draw_detection_bounding_boxes(self, frame):
        if self.detector.detections is not None:
            colour = np.asarray((0., 0., 255.))
            for single_detection in self.detector.detections:
                top_left = torch.unbind(single_detection[:2].int())
                bottom_right = torch.unbind(single_detection[2:4].int())
                draw_rectangle(frame, top_left, bottom_right, colour)

    def deep_sort_loop(self, frame):
        tlwh_detections, obj_confidences = reformat_detections(self.detector.detections)
        tracker, _ = self.pedestrian_tracker.run_deep_sort(frame, obj_confidences, tlwh_detections)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 2:
                continue
            tracked_bbox = track.to_tlbr()
            track_id = track.track_id
            colour = (200, 0, 0)
            draw_rectangle(frame, (int(tracked_bbox[0]), int(tracked_bbox[1])), (int(tracked_bbox[2]), int(tracked_bbox[3])), colour)

    def calculate_pairwise_distance(self, tracks):
        pass


def display_image(window_title, frame):
    cv2.imshow(window_title, frame)


def draw_rectangle(frame, top_left_pos, bottom_right_pos, colour):
    cv2.rectangle(img=frame, pt1=top_left_pos, pt2=bottom_right_pos, color=colour)


def draw_line(frame, start_pos, end_pos, colour):
    cv2.line(img=frame, pt1=start_pos, pt2=end_pos, color=colour)


def draw_text(frame, text, position, scale, colour):
    cv2.putText(img=frame, text=text, org=position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale, color=colour)


if __name__ == '__main__':
    args = parse_command_line_args()
    capture = cv2.VideoCapture(args.input_path)
    observer = PedestrianObserver(args, capture)
    observer.social_distance_loop()




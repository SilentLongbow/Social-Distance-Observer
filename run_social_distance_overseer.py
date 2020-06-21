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

RED = (0., 0., 255.)
WHITE = (255., 255., 255.)


class PedestrianObserver:

    def __init__(self, args, capture):
        self.args = args
        self.capture = capture
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output = get_output(capture)

        self.detector = detector.PedestrianDetector(self.args.model_definition,
                                                    self.args.weights_path,
                                                    self.args.image_size,
                                                    self.args.class_path,
                                                    self.device)

        # Add the deep_sort_python directory to sys as the pre-trained model needs deep_sort_python as working directory
        sys.path.append("deep_sort_python")
        self.pedestrian_tracker = deepsort.deepsort_rbc()
        sys.path.remove("deep_sort_python")

        self.ids_at_risk = set()
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.confidence_threshold = args.conf_threshold
        self.iou_threshold = args.non_max_supp_threshold
        self.min_distance = args.min_distance

        self.start_time = None
        self.end_time = None
        self.frames_processed = 0

    def social_distance_loop(self):
        """
        Reads each frame in the opened capture, firing off calls to the object detector first.
        One the detector has completed, it launches the object tracker.
        Once the objects have been tracked, the distance calculations are performed.
        """
        self.start_time = datetime.now()
        while self.capture.isOpened():
            retrieved, frame = capture.read()
            if not retrieved:
                break
            self.frames_processed += 1
            self.perform_object_detection(frame)
            if self.args.track_detections:
                self.deep_sort_loop(frame)
            else:
                self.draw_detection_bounding_boxes(frame)
            display_image("Social Distancing", frame)
            self.output.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.process_runtime_statistics()

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
        tracker, detections = self.pedestrian_tracker.run_deep_sort(frame, obj_confidences, tlwh_detections)
        for track in tracker.tracks:
            if not track.is_confirmed() and track.time_since_update < 2:
                continue
            track_bbox = track.to_tlbr()
            track_id = track.track_id
            track_top_left = tuple(track_bbox[:2].astype(int))
            track_bottom_right = tuple(track_bbox[2:4].astype(int))
            box_colour = WHITE
            if not self.is_social_distancing(frame, track, tracker.tracks):
                self.ids_at_risk.add(track_id)
                box_colour = RED
            draw_rectangle(frame, track_top_left, track_bottom_right, box_colour)
            draw_text(frame, str(track_id), track_top_left, 5e-3 * 100, RED)
        self.display_at_risk_text(frame)

    def is_social_distancing(self, frame, current_track, tracks):
        is_social_distancing = True
        if len(tracks) >= 2:
            for other_track in tracks:
                if (current_track.track_id == other_track.track_id) or not other_track.is_confirmed():
                    continue
                current_track_centroid = current_track.to_x0y0()
                other_track_centroid = other_track.to_x0y0()
                two_d_track_distance = np.linalg.norm(current_track_centroid - other_track_centroid)
                too_close = two_d_track_distance < self.min_distance
                if too_close:
                    is_social_distancing = False
                    draw_line(frame,
                              tuple(current_track_centroid.astype(int)),
                              tuple(other_track_centroid.astype(int)),
                              RED)
        return is_social_distancing

    def display_at_risk_text(self, frame):
        at_risk_text = "Total at risk: {}".format(len(self.ids_at_risk))
        text_position = (10, frame.shape[0] - 10)
        draw_text(frame, at_risk_text, text_position, 2, RED)

    def process_runtime_statistics(self):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).seconds
        duration_text = "Time taken: {}s".format(duration)
        frames_text = "Frames processed: {}".format(self.frames_processed)
        average_rate_text = "Average framerate: {:.2f}fps".format(self.frames_processed / float(duration))

        print(duration_text)
        print(frames_text)
        print(average_rate_text)


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
    parser.add_argument("--detect-and-track", dest='track_detections', action='store_true',
                        help="perform tracking on top of YOLOv3 detections")
    parser.add_argument("--detect-only", dest='track_detections', action='store_false',
                        help="perform only YOLOv3 detection")
    parser.set_defaults(track_detections=False)
    return parser.parse_args()


def get_output(capture):
    filename = "video_output/detections-{}.avi".format(datetime.now().time().minute)
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


def display_image(window_title, frame):
    cv2.imshow(window_title, frame)


def draw_rectangle(frame, top_left_pos, bottom_right_pos, colour):
    cv2.rectangle(img=frame, pt1=top_left_pos, pt2=bottom_right_pos, color=colour)


def draw_line(frame, start_pos, end_pos, colour):
    cv2.line(img=frame, pt1=start_pos, pt2=end_pos, color=colour)


def draw_text(frame, text, position, scale, colour):
    cv2.putText(img=frame, text=text, org=position, fontFace=1, fontScale=scale, color=colour)


if __name__ == '__main__':
    args = parse_command_line_args()
    capture = cv2.VideoCapture(args.input_path)
    observer = PedestrianObserver(args, capture)
    observer.social_distance_loop()




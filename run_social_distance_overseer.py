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
            self.display_image("Social Distancing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def perform_object_detection(self, frame):
        detection_image = detector.convert_image_for_detection(frame, self.args.image_size, self.Tensor)
        self.detector.perform_pedestrian_detection(detection_image, self.confidence_threshold, self.iou_threshold)
        if self.detector.detections is not None:
            self.detector.restore_bounding_boxes(self.args.image_size, frame.shape[:2])
            self.detector.cull_non_pedestrian_detections()
            # TODO Handle when all non-person objects are culled.
            for x1, y1, x2, y2, obj_conf, class_conf, class_pred in self.detector.detections:
                colour = (0., 0., 255.)
                self.draw_rectangle(frame, (x1, y1), (x2, y2), colour)

    def deep_sort_loop(self):
        pass

    def display_image(self, window_title, frame):
        cv2.imshow(window_title, frame)

    def draw_rectangle(self, frame, top_left_post, bottom_right_pos, colour):
        cv2.rectangle(img=frame, pt1=top_left_post, pt2=bottom_right_pos, color=colour)

    def draw_line(self, frame, start_pos, end_pos, colour):
        cv2.line(img=frame, pt1=start_pos, pt2=end_pos, color=colour)

    def draw_text(self, frame, text, position, scale, colour):
        cv2.putText(img=frame,text=text, org=position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale, color=colour)




if __name__ == '__main__':
    args = parse_command_line_args()

    capture = cv2.VideoCapture(args.input_path)
    # capture = cv2.VideoCapture(0)
    observer = PedestrianObserver(args, capture)
    observer.social_distance_loop()




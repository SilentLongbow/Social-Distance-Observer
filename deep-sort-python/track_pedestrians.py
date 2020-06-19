import torchvision
import torchvision.datasets
import torchvision.transforms as transforms

import yolov3.detect_pedestrians as ped_dect

from deepsort import *


class PedestrianTracker:
    def __init__(self):
        self.deepsort = deepsort_rbc()



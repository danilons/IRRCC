# coding: utf-8
import os
import itertools
import h5py
import numpy as np
import collections
from fast_rcnn.nms_wrapper import nms
from relation import RCC


class Dataset(object):

    def __init__(self, path, suffix='train'):
        self.coordinates = h5py.File(os.path.join(path, 'dataset_{}.hdf5'.format(suffix)))
        self.objects = h5py.File(os.path.join(path, 'objects_{}.hdf5'.format(suffix)))

        self.classes = collections.OrderedDict()
        with open(os.path.join(path, 'names.txt'), 'r') as fp:
            for line in fp.readlines():
                index, name = line.strip().split()
                self.classes[int(index)] = name

        # topological stuff
        self.detector = RCC()

    @property
    def images(self):
        return self.coordinates.keys()

    def boxes(self, image):
        return self.objects[image]['boxes']

    def scores(self, image):
        return self.objects[image]['scores']

    def get_objects(self, image, confidence=0.8, nox_supression_max=0.3, only_with_objects=True):
        scores = self.scores(image)
        boxes = self.boxes(image)

        detections = {}
        for k in self.classes:
            indices = np.where(scores[:, k] > confidence)[0]
            print((indices, indices.dtype))
            class_scores = scores[indices, k]
            class_boxes = boxes[indices, k * 4: (k + 1) * 4]
            class_detections = np.hstack((class_boxes, class_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(class_detections, nox_supression_max)
            class_detections = class_detections[keep, :]
            # verify it there is something detected
            if only_with_objects and len(class_detections) == 0:
                continue

            detections[k] = class_detections

        return detections

    def topology_relation(self, image):
        objects = self.get_objects(image)
        boxes = np.array(objects.values())
        relations = []

        for box1, box2 in itertools.permutations(boxes, boxes):
            x1, y1, x2, y2 = box1[:4].astype(np.int32)
            contour1 = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])

            x1, y1, x2, y2 = box2[:4].astype(np.int32)
            contour2 = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])

            relation = self.detector.compute(image, contour1, contour2)
            relations.append([box1, box2, relation.scope])

        return relations

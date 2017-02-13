# coding: utf-8
import os
import itertools
import h5py
import cv2
import numpy as np
import collections
from scipy.io import loadmat
from utils import nms
from relation import RCC


class Dataset(object):

    def __init__(self, path, suffix='train', image_path='images'):
        self.coordinates = h5py.File(os.path.join(path, 'dataset_{}.hdf5'.format(suffix)))
        self._detector = Detection(path=path, suffix=suffix)
        self._segmentation = Segmentation(path=path, suffix=suffix)
        self.image_path = image_path

    @property
    def detector(self):
        return self._detector

    @property
    def segmentation(self):
        return self._segmentation

    @property
    def images(self):
        return self.coordinates.keys()

    def ground_truth(self, image):
        gold_standard = self.coordinates[image]
        contour = {}
        for classname in gold_standard:
            bbox = gold_standard.get(classname)
            contour[classname] = np.vstack((bbox['x'], bbox['y'])).T
        return contour

    def get_im_array(self, image, rgb=False):
        if rgb:
            cv2.imread(os.path.join(self.image_path, image))[:, :, (2, 1, 0)]
        return cv2.imread(os.path.join(self.image_path, image))     
        
    def get_image_with_objects(self, image, obj_id=None, **kwargs):
        img = self.get_im_array(image, **kwargs)
        self.detector.get_image_with_objects(img, image, obj_id, **kwargs)
        return img

class Detection(object):

    def __init__(self, path, suffix='train'):
        filename = os.path.join(path, 'objects_{}.hdf5'.format(suffix))
        print("Reading objects from: {}".format(filename))
        self.objects = h5py.File(filename)
        self._classes = collections.OrderedDict()
        with open(os.path.join(path, 'rcnn' , 'names.txt'), 'r') as fp:
            for line in fp.readlines():
                index, name = line.strip().split()
                self._classes[np.int32(index)] = name

        self._classname = {v: k for k, v in self._classes.iteritems()}
        # topological stuff
        self.detector = RCC()

    @property
    def classes(self):
        return self._classes

    def get_object_id(self, classname):
        return self._classname.get(classname, 0)

    def boxes(self, image):
        return np.array(self.objects[image]['boxes'])

    def scores(self, image):
        return np.array(self.objects[image]['scores'])

    def get_objects(self, image, **kwargs):
        scores = self.scores(image)
        boxes = self.boxes(image)
       
        confidence = kwargs.get('confidence', 0.8)
        nox_supression_max = kwargs.get('nox_supression_max', 0.3)
        only_with_objects = kwargs.get('only_with_objects')

        detections = {}
        for k in self.classes:
            if k == 0:  # background
                continue 

            indices = np.where(scores[:, k] > confidence)[0]
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

    def get_image_with_objects(self, img, image, obj_id=None, **kwargs):
        colors = kwargs.get('colors', {})
        objects = self.get_objects(image)
        objects = objects if not obj_id else {obj_id: objects.get(obj_id, [])}

        for k in objects:
            for detection in objects[k]:
                bbox = detection[:4].astype(np.int32)
                score = detection[-1]
                class_name = self.classes[k]
                color = colors.get(k, (0, 0, 255))
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(img, '{:s} {:.3f}'.format(class_name, score), 
                            (bbox[0], bbox[1] + 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return img


    def get_image_with_box_pair(self, image, box1, box2):
        img = image.copy()
        color = (0, 0, 255)
        cv2.rectangle(img, (box1[0], box1[1]), (box1[2], box1[3]), color, 2)
        cv2.rectangle(img, (box2[0], box2[1]), (box2[2], box2[3]), color, 2)
        return img

    def topology_relation(self, shape,  image):
        objects = self.get_objects(image)
        if len(objects) == 0:
            return []

        boxes = np.vstack(objects.values())
        if len(objects) == 1:
            return [(boxes, boxes, 'EQ')]

        relations = []
        nn1 = 0
        for box1 in boxes:
            nn1 += 1
            for box2 in boxes[nn1:]:
                x1, y1, x2, y2 = box1[:4].astype(np.int32)
                contour1 = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])

                x1, y1, x2, y2 = box2[:4].astype(np.int32)
                contour2 = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])

                relation = self.detector.compute(shape, contour1, contour2)
                relations.append((box1, box2, relation.scope))

        return relations

class Segmentation(Detection):
    def __init__(self, path, suffix='train'):
        fname = os.path.join(path, 'segmentation_{}.hdf5'.format(suffix))
        print("Reading segmentation from {}".format(fname))
        self.objects = h5py.File(fname)
        self._classes = collections.OrderedDict()
        with open(os.path.join(path, 'sceneparsing' , 'names.txt'), 'r') as fp:
            for line in fp.readlines()[1:]:
                index, _, _, _, name = line.strip().split('\t')
                self._classes[np.int32(index)] = name

        self._classname = {v: k for k, v in self._classes.iteritems()}
        # topological stuff
        self.detector = RCC()

    def get_objects(self, image, **kwargs):
        objects = np.array(self.objects[image])
        classes = np.unique(objects)
        segmentation = {}

        for k in classes:
            x, y = np.where(objects == k)
            img = np.zeros(objects.shape[:2], dtype=np.uint8)
            img[x, y] = 255.
            _, binary = cv2.threshold(img, 127,255,0)
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            biggest = sorted([(len(cnt), nn) for nn, cnt in enumerate(contours)], key=lambda x: x[0], reverse=True)
            _, idx = biggest[0]
            segmentation[self.classes[k]] = contours[idx]

        return segmentation

    def topology_relation(self, shape,  image):
        objects = self.get_objects(image)
        if len(objects) == 0:
            return []

        contours = objects.items()
        if len(objects) == 1:
            return [(contours, contours, 'EQ')]

        relations = []
        nn1 = 0
        for obj1, c1 in contours:
            nn1 += 1
            x, y, w, h = cv2.boundingRect(c1)
            x1, y1, x2, y2 = x, y, x + w, y + h
            contour1 = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])

            for obj2, c2 in contours[nn1:]:
                x, y, w, h = cv2.boundingRect(c2)
                x1, y1, x2, y2 = x, y, x + w, y + h
                contour2 = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])

                relation = self.detector.compute(shape, contour1, contour2)
                relations.append({'objects': (self.classes[obj1], self.classes[obj2]), 
                                  'contours': (contour1, contour2), 
                                  'relation': relation.scope})

        return relations

    def get_object_names(self, image, **kwargs):
        return [self.classes[obj] for obj in self.get_objects(image)]



# coding: utf-8
import os
import itertools
import h5py
import cv2
import numpy as np
import collections
from scipy.io import loadmat
from fast_rcnn.nms_wrapper import nms
from relation import RCC


class Dataset(object):

    def __init__(self, path, suffix='train', image_path='images'):
        self.coordinates = h5py.File(os.path.join(path, 'dataset_{}.hdf5'.format(suffix)))
        self.objects = h5py.File(os.path.join(path, 'objects_{}.hdf5'.format(suffix)))

        self._classes = collections.OrderedDict()
        with open(os.path.join(path, 'names.txt'), 'r') as fp:
            for line in fp.readlines():
                index, name = line.strip().split()
                self._classes[np.int32(index)] = name

        self._classname = {v: k for k, v in self._classes.iteritems()}
        self.image_path = image_path
        # topological stuff
        self.detector = RCC()

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

    @property
    def classes(self):
        return self._classes

    def get_object_id(self, classname):
        return self._classname.get(classname, 0)

    def get_im_array(self, image, rgb=False):
        if rgb:
            cv2.imread(os.path.join(self.image_path, image))[:, :, (2, 1, 0)]
        return cv2.imread(os.path.join(self.image_path, image))


    def boxes(self, image):
        return np.array(self.objects[image]['boxes'])

    def scores(self, image):
        return np.array(self.objects[image]['scores'])

    def get_objects(self, image, confidence=0.8, nox_supression_max=0.3, only_with_objects=True):
        scores = self.scores(image)
        boxes = self.boxes(image)
       
        detections = {}
        for k in self.classes:
            if k == 0:  # background
                continue 

            indices = np.where(scores[:, k] > confidence)[0]
            class_scores = scores[indices, k]
            class_boxes = boxes[indices, k * 4: (k + 1) * 4]
            class_detections = np.hstack((class_boxes, class_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(class_detections, nox_supression_max, force_cpu=True)
            class_detections = class_detections[keep, :]
            # verify it there is something detected
            if only_with_objects and len(class_detections) == 0:
                continue

            detections[k] = class_detections

        return detections

    def topology_relation(self, image):
        objects = self.get_objects(image)
        if len(objects) == 0:
            return []

        boxes = np.vstack(objects.values())
        if len(objects) == 1:
            return [(boxes, boxes, 'EQ')]

        relations = []
        im = self.get_im_array(image)
        nn1 = 0
        for box1 in boxes:
            nn1 += 1
            for box2 in boxes[nn1:]:
                x1, y1, x2, y2 = box1[:4].astype(np.int32)
                contour1 = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])

                x1, y1, x2, y2 = box2[:4].astype(np.int32)
                contour2 = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])

                relation = self.detector.compute(im, contour1, contour2)
                relations.append((box1, box2, relation.scope))

        return relations

    def get_image_with_objects(self, image, obj_id=None, **kwargs):
        img = self.get_im_array(image, **kwargs)
        
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
        img = self.get_im_array(image, **kwargs)        
        color = (0, 0, 255)
        cv2.rectangle(img, (box1[0], box1[1]), (box1[2], box1[3]), color, 2)
        cv2.rectangle(img, (box2[0], box2[1]), (box2[2], box2[3]), color, 2)
        return img


class StructuredQuery:
    def __init__(self, fname):
        self.db = loadmat(fname, struct_as_record=True, chars_as_strings=True, squeeze_me=True)
        valid_queries = [q for q in self.query if len(np.nonzero(q['rank'])[0].tolist()) > 30]
        
        self.queries = {}
        self.queries['a'] = [dict(zip(q.dtype.names, q)) for q in valid_queries 
                             if isinstance(q['unary'], np.ndarray) and q['unary'].size == 0 and q['binary'].size == 3]
        self.queries['b'] = [dict(zip(q.dtype.names, q)) for q in self.query if q['unary'] != 0 and q['binary'].size == 3]
        self.queries['c'] = [dict(zip(q.dtype.names, q)) for q in self.query 
                             if isinstance(q['unary'], np.ndarray) and q['unary'].size == 0 and q['binary'].size == 6]
        self.queries['d'] = [dict(zip(q.dtype.names, q)) for q in self.query if q['unary'] != 0 and q['binary'].size == 6]
        self.queries['e'] = [dict(zip(q.dtype.names, q)) for q in self.query 
                             if isinstance(q['unary'], np.ndarray) and q['unary'].size == 0 and q['binary'].size == 9]
        
    @property
    def names(self):
        return self.db['names']
    
    @property
    def relations(self):
        return self.db['relations']
    
    @property
    def query(self):
        return self.db['Query']
    
    @property
    def query_types(self):
        return self.queries.keys()
    
    def __getitem__(self, query_type):
        queries = []
        for query in self.queries.get(query_type, []):
            unary = None
            if query['unary']:
                unary = structured_queries.names[query['unary'] - 1]
            try:
                name1 = structured_queries.names[query['binary'][:, 0].squeeze() - 1]
                prepo = structured_queries.relations[query['binary'][:, 2].squeeze() - 1]
                name2 = structured_queries.names[query['binary'][:, 1].squeeze() - 1]
            except IndexError:
                name1 = structured_queries.names[query['binary'][0].squeeze() - 1]
                prepo = structured_queries.relations[query['binary'][2].squeeze() - 1]
                name2 = structured_queries.names[query['binary'][1].squeeze() - 1]
            
            query_name = np.vstack((name1, prepo, name2)).reshape(3, -1).T
            if unary:
                name = unary + ", " + " & ".join(["-".join(q) for q in query_name])
            else:
                name = " & ".join(["-".join(q) for q in query_name])
            queries.append({'binary': query['binary'] - 1, 
                            'unary': query['unary'] - 1, 
                            'name': name,
                            'rank': query['rank'].astype(bool)
                           })
        return queries

class QueryAnnotation:
    def __init__(self, path):
        self.db = {}
        for folder in glob(os.path.join(anno_path, '*')):
            for fname in glob(os.path.join(folder, '*.txt')):
                key = os.path.basename(folder) 
                with open(fname, 'r') as fp:
                    self.db[key] = [line.strip() for line in fp.readlines() if line.strip().endswith('.jpg')]
        
        names1, preposition, names2 = zip(*(query.split('-') for query in self.db))
        self.names = list(set(names1) | set(names2))
        self.preposition = list(set(preposition))
        
    def __getitem__(self, term):
        return self.db.get(term, [])

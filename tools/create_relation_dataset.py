# coding: utf-8
import os
import sys
path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, path)
import argparse
import pandas as pd
import numpy as np
import cv2
from irrcc.dataset import Dataset
from irrcc.query_annotation import QueryAnnotation


def create_relation(handler, images, noun1, noun2, **kwargs):
    objects = handler.get_objects(images)

    if kwargs.get('segmentation') == 'segmentation':
        objects = {handler._classname[k]: v for k, v in objects.items()}
        cnt1 = objects.get(noun1)
        cnt2 = objects.get(noun2)

        if cnt1 is None or cnt2 is None:
            return ''

        x1, y1, w1, h1 = cv2.boundingRect(cnt1)
        x11, y11, x12, y12 = x1, y1, x1 + w1, y1 + h1
        x11, x12 = sorted([x11, x12])
        y11, y12 = sorted([y11, y12])
        contour1 = np.array([[x11, y11], [x11, y12], [x12, y12], [x12, y11]])

        x2, y2, w2, h2 = cv2.boundingRect(cnt2)
        x21, y21, x22, y22 = x2, y2, x2 + w2, y2 + h2
        x21, x22 = sorted([x21, x22])
        y21, y22 = sorted([y21, y22])
        contour2 = np.array([[x21, y21], [x21, y22], [x22, y22], [x22, y21]])

    else:
        objects = {handler.classes[k]: v for k, v in objects.items()}
        cnt1 = objects.get(noun1)
        cnt2 = objects.get(noun2)
        if cnt1 is None or cnt2 is None:
            return ''

        if cnt1.size <= 0 or cnt2.size <= 0:
            return ''

        cnt1 = cnt1.flatten()
        cnt2 = cnt2.flatten()

        x1, y1, x2, y2 = cnt1[:4].astype(np.int32)
        contour1 = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])

        x1, y1, x2, y2 = cnt2[:4].astype(np.int32)
        contour2 = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])

    return handler.detector.compute((384, 384), contour1, contour2).scope

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/')
    parser.add_argument('-i', '--image_path', action="store", default='images')
    parser.add_argument('--segmentation', dest='segmentation', action="store_true", default=True)
    parser.add_argument('--no-segmentation', dest='segmentation', action='store_false')
    parser.set_defaults(feature=True)
    params = parser.parse_args()

    segmentation = 'segmentation' if params.segmentation else 'detector'
    print("Processing {}".format(segmentation))
        
    for mode in ['train', 'test']:
        print("Processing file {}".format(mode))
        location = os.path.join(params.dataset_path, '{}_anno'.format(mode.lower()))
        query_annotation = QueryAnnotation(location)
        dset = Dataset(params.dataset_path, mode, params.image_path)
        
        df = pd.DataFrame.from_records(query_annotation.features(), columns=['images', 'noun1', 'preposition', 'noun2'])
        print("Query annotated imgs {}".format(len(query_annotation.imgs)))
        print("Records to process: {}".format(len(df)))
        
        handler = dset.segmentation if params.segmentation else dset.detector

        df['rcc'] = df.apply(lambda x: create_relation(handler=handler, segmentation=segmentation, **x), axis=1)
        print("Processed records {}".format(df.shape))
        
        fname = os.path.join(params.dataset_path, segmentation, mode + '_relation.csv')
        print("Saving file to: {}".format(fname))

        df.to_csv(fname, index=None)

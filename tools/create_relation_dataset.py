# coding: utf-8
import os
import sys
path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, path)
import argparse
import pandas as pd
import numpy as np
import cv2
import caffe
from skimage import img_as_ubyte
from skimage.transform import resize
from irrcc.dataset import Dataset
from irrcc.query_annotation import QueryAnnotation


def create_relation(handler, images, noun1, noun2, **kwargs):
    objects = handler.get_objects(images)
    objects = {handler.classes[k]: v for k, v in objects.items()}
    cnt1 = objects.get(noun1)
    cnt2 = objects.get(noun2)
    
    if cnt1 is None or cnt2 is None:
        return ''
    
    x, y, w, h = cv2.boundingRect(cnt1)
    x1, y1, x2, y2 = x, y, x + w, y + h
    contour1 = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
    
    x, y, w, h = cv2.boundingRect(cnt2)
    x1, y1, x2, y2 = x, y, x + w, y + h
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

    for mode in ['train', 'test']:
        print("Processing file {}".format(mode))
        location = os.path.join(params.dataset_path, '{}_anno'.format(mode.lower()))
        query_annotation = QueryAnnotation(location)
        dset = Dataset(params.dataset_path, mode, params.image_path)
        
        df = pd.DataFrame.from_records(query_annotation.features(), columns=['images', 'noun1', 'preposition', 'noun2'])
        print("Query annotated imgs {}".format(len(query_annotation.imgs)))
        print("Records to process: {}".format(len(df)))
        
        handler = dset.segmentation if params.segmentation else dset.detector

        df['rcc'] = df.apply(lambda x: create_relation(handler=handler, **x), axis=1)
        df['predicted_noun1'] = df.apply(dset.images.get())

        segmentation = 'segmentation' if params.segmentation else 'detector'
        fname = os.path.join(params.dataset_path, 'segmentation', mode + '_relation.csv')
        print("Saving file to: {}".format(fname))

        print("Processed records {}".format(df.shape))
        df.to_csv(fname, index=None)

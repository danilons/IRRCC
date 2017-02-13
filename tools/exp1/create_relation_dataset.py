# coding: utf-8
import os
import sys
import pandas as pd
import argparse
import json
path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, path)
import click
import cv2
import numpy as np
from irrcc.dataset import Dataset
from irrcc.structured_query import StructuredQuery


def create_relation(handler, images, noun1, noun2, **kwargs):
    try:
        objects = handler.get_objects(images)
    except KeyError:
        return ''

    if kwargs.get('segmentation') == 'segmentation':
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

    else:
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

def create_dataset(sq, imagenames):
    data = {}
    imagenames = np.array(imagenames)
    with click.progressbar(length=len(sq.query), show_pos=True, show_percent=True) as bar:
        for query_type in sq.query_types:
            for sq_type in sq[query_type]:
                try:
                    unary = sq.names[sq_type['unary']]
                except IndexError:
                    unary = ''
                for query_term in sq_type['name'].split(','):
                    for term in query_term.split('&'):
                        try:
                            noun1, prep, noun2 = term.split('-')
                        except ValueError:
                            continue

                        images = imagenames[sq_type['rank']]
                        for img in images:
                            data.setdefault('type', []).append(query_type)
                            data.setdefault('images', []).append(img)
                            data.setdefault('noun1', []).append(noun1.strip())
                            data.setdefault('noun2', []).append(noun2.strip())
                            data.setdefault('preposition', []).append(prep)
                            data.setdefault('unary', []).append(unary)
                bar.update(1)

        return pd.DataFrame(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/')
    parser.add_argument('-i', '--image_path', action="store", default='images')
    parser.add_argument('-o', '--output_path', action="store", default='data/exp1')
    params = parser.parse_args()


    for trainset in ['test']:
        sq = StructuredQuery(os.path.join(params.dataset_path,'Struct-Query-{}.mat'.format(trainset.title())))
        with open(os.path.join(params.dataset_path, 'imagenames_{}.json'.format(trainset)), 'r') as fp:
            imagenames = json.load(fp)

        df = create_dataset(sq, imagenames)
        
        dset = Dataset(params.dataset_path, trainset, params.image_path)
        df = df[df['type'] == 'a'].drop_duplicates()
        df['rcc'] = df.apply(lambda x: create_relation(handler=dset.segmentation, segmentation='segmentation', **x), axis=1)
        df.to_csv(os.path.join(params.output_path, trainset + '.csv'), index=None)

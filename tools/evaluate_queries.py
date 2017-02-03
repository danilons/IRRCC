# coding: utf-8
import os
import argparse
import click
import numpy as np
import pandas as pd
import sys
path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, path)
from irrcc.dataset import Dataset
from irrcc.query_annotation import QueryAnnotation

def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data')
    parser.add_argument('--segmentation', dest='segmentation', action="store_true", default=True)
    parser.add_argument('--no-segmentation', dest='segmentation', action='store_false')
    parser.set_defaults(feature=True)
    params = parser.parse_args()

    segmentation = 'segmentation' if params.segmentation else 'detector'

    location = os.path.join(params.dataset_path, 'test_anno')
    qa = QueryAnnotation(location)
    df = pd.read_csv(os.path.join(params.dataset_path, segmentation, 'predicted_relation.csv'))

    print("Intersection size is {}".format(len(set(df.images) & set(qa.imgs))))
    print("QA imgs", len(set(qa.imgs)))
    print("DB imgs", len(set(df.images)))

    assert len(set(df.images) & set(qa.imgs)) == len(qa.imgs), "Number of images different with query annotation"
    assert len(set(df.images) & set(qa.imgs)) == len(set(df.images)), "Number of images different with dataset"

    mean_average_precision = []
    with click.progressbar(length=len(qa.db), show_pos=True, show_percent=True) as bar:
        for query in qa.db:
            noun1, preposition, noun2 = query.split('-')
            retrieved = df[(df['noun1'] == noun1) & (df['preposition'] == preposition) & (df['noun2'] == noun2) & (df['rcc'].notnull())].images
            score = apk(qa.db[query], retrieved)
            mean_average_precision.append(score)
            bar.update(1)

    print("mAP {:.4f}".format(np.mean(mean_average_precision)))


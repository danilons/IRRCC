# coding: utf-8
from __future__ import division
import os
import json
import argparse
import click
import numpy as np
import pandas as pd
import cytoolz
import json
import sys
from sklearn.metrics import average_precision_score, roc_curve
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
    parser.add_argument('-i', '--image_list', action="store", default='data/imagenames_test.json')
    parser.add_argument('-q', '--queries_path', action="store", default='data/segmentation/query_equivalence.csv')
    parser.add_argument('-t', '--threshold', action="store", default=3.0, type=float)
    parser.add_argument('--segmentation', dest='segmentation', action="store_true", default=True)
    parser.add_argument('--no-segmentation', dest='segmentation', action='store_false')
    parser.set_defaults(feature=True)
    parser.add_argument('--equivalents', dest='equivalents', action="store_true", default=True)
    parser.add_argument('--no-equivalents', dest='equivalents', action='store_false')
    parser.set_defaults(feature=True)
    params = parser.parse_args()

    segmentation = 'segmentation' if params.segmentation else 'detector'
    print("Processing {}".format(segmentation))

    with open(params.image_list, 'r') as fp:
        imagenames = json.load(fp)

    location = os.path.join(params.dataset_path, 'test_anno')
    qa = QueryAnnotation(location)
    df = pd.read_csv(os.path.join(params.dataset_path, segmentation, 'predicted_relation.csv'))

    queries = pd.read_csv(params.queries_path)
    queries = dict(zip(queries['Original'], queries['Equivalent']))

    print("Intersection size is {}".format(len(set(df.images) & set(qa.imgs))))
    print("QA imgs", len(set(qa.imgs)))
    print("DB imgs", len(set(df.images)))

    assert len(set(df.images) & set(qa.imgs)) == len(qa.imgs), "Number of images different with query annotation"
    assert len(set(df.images) & set(qa.imgs)) == len(set(df.images)), "Number of images different with dataset"
    
    weights = df.images.value_counts().to_dict()
    negative = len(imagenames)
    avg_precision = []
    mean_average_precision = []
    
    with click.progressbar(length=len(qa.db), show_pos=True, show_percent=True) as bar:
        for nn, query in enumerate(qa.db):
            if params.equivalents:
                equivalent = queries.get(query)
                try:
                    if np.isnan(equivalent) or equivalent is None:
                        print("Skipping {}".format(query))
                        continue
                except TypeError:
                    pass

            noun1, preposition, noun2 = query.split('-')
            l1 = list(df[(df['noun1'] == noun1) & (df['rcc'].notnull())].images)
            l2 = list(df[(df['noun2'] == noun2) & (df['rcc'].notnull())].images)
            l3 = list(df[(df['preposition'] == preposition) & (df['rcc'].notnull())].images)

            retrieved = {k: v / weights[k] for k, v in cytoolz.frequencies(l1 + l2 + l3).items()}
            valids = [(k, retrieved[k]) for k in sorted(retrieved, key=retrieved.get, reverse=True) if retrieved[k] >= params.threshold]
            retrieved = []
            relevance = []
            if valids:
                retrieved, relevance  = zip(*valids)

            tp = len(set(qa.db[query]) & set(retrieved))
            fp = len(retrieved) - tp
            tn = negative - len(set(qa.db[query]))
            fn = len(qa.db[query]) - tp 

            y_test = np.zeros(negative, dtype=np.float32)
            for img in qa.db[query]:
                y_test[imagenames.index(img)] = 1.

            y_scores = np.zeros(negative, dtype=np.float32)
            for nn, img in enumerate(retrieved):
                y_scores[imagenames.index(img)] = relevance[nn]

            average_precision = average_precision_score(y_test, y_scores)
            if np.isnan(average_precision):
                avg_precision.append(0.0)
            else:
                avg_precision.append(average_precision)
            # retrieved = df[(df['noun1'] == noun1) & (df['preposition'] == preposition) & (df['noun2'] == noun2) & (df['rcc'].notnull())].images
            # score = apk(qa.db[query], retrieved)
            mean_average_precision.append({'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 
                                           'y_test': y_test.tolist(), 
                                           'y_scores': y_scores.tolist()})
            bar.update(1)

    with open(os.path.join(params.dataset_path, segmentation, 'map.json'), 'w') as fp:
        json.dump(mean_average_precision, fp)
    print("mAP {:.4f}".format(np.mean(avg_precision)))


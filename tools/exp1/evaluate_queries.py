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
from sklearn.metrics import average_precision_score
path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, path)
from irrcc.dataset import Dataset
from irrcc.structured_query import StructuredQuery

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
    parser.add_argument('-t', '--threshold', action="store", default=3.0, type=float)
    parser.add_argument('--segmentation', dest='segmentation', action="store_true", default=True)
    parser.add_argument('--no-segmentation', dest='segmentation', action='store_false')
    parser.set_defaults(feature=True)
    params = parser.parse_args()

    with open(params.image_list, 'r') as fp:
        imagenames = json.load(fp)

    sq = StructuredQuery(os.path.join(params.dataset_path, 'Struct-Query-Test.mat'))
    df = pd.read_csv(os.path.join(params.dataset_path, 'exp1', 'predicted_relation.csv'))
    
    map_ = {}
    weights = df.images.value_counts().to_dict()
    negative = len(imagenames)
    
    for query_type in sorted(sq.query_types):
        avg_precision = []
        mean_average_precision = []
        with click.progressbar(length=len(sq[query_type]), show_pos=True, show_percent=True) as bar:
            for query in sq[query_type]:
                l1 = []
                l2 = []
                l3 = []
                l4 = []
                for term in query['name'].split('&'):
                    try:
                        unary, binary = term.split(',')
                    except ValueError:
                        binary = term
                        unary = ''

                    noun1, preposition, noun2 = binary.split('-')
                    l1 += list(df[(df['noun1'] == noun1) & (df['rcc'].notnull())].images)
                    l2 += list(df[(df['noun2'] == noun2) & (df['rcc'].notnull())].images)
                    l3 += list(df[(df['preposition'] == preposition) & (df['rcc'].notnull())].images)
                    l4 += list(df[((df['noun1'] == unary) | (df['noun2'] == unary)) & (df['rcc'] == 'DC')].images)
                
                retrieved = {k: v / weights[k] for k, v in cytoolz.frequencies(l1 + l2 + l3 + l4).items()}
                valids = [(k, retrieved[k]) for k in sorted(retrieved, key=retrieved.get, reverse=True) if retrieved[k] >= 2.5]
                retrieved = []
                relevance = []
                if valids:
                    retrieved, relevance  = zip(*valids)
                
                gs = [imagenames[idx] for idx, is_valid in enumerate(query['rank']) if is_valid]
                tp = len(set(gs) & set(retrieved))
                fp = len(retrieved) - tp
                tn = negative - len(set(gs))
                fn = len(gs) - tp 

                y_test = query['rank'].astype(np.float32)

                y_scores = np.zeros(negative, dtype=np.float32)
                for nn, img in enumerate(retrieved):
                    y_scores[imagenames.index(img)] = relevance[nn]

                average_precision = average_precision_score(y_test, y_scores)
                if np.isnan(average_precision):
                    avg_precision.append(0.0)
                else:
                    avg_precision.append(average_precision)

                mean_average_precision.append({'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 
                                               'y_test': y_test.tolist(), 
                                               'y_scores': y_scores.tolist()})

                map_[query_type] = mean_average_precision
                bar.update(1)
            print("\nQuery {} mAP {:.4f}".format(query_type, np.mean(avg_precision)))
                
    with open(os.path.join(params.dataset_path, 'exp1', 'map.json'), 'w') as fp:
        json.dump(map_, fp)
    

# coding: utf-8
import os
import sys
import joblib
import json
path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, path)
import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, average_precision_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/exp1')
    parser.set_defaults(feature=True)
    params = parser.parse_args()

    filename = os.path.join(params.dataset_path, 'train.csv')
   
    df = pd.read_csv(filename)
    df.fillna('', inplace=True)
    df[['noun1', 'noun2', 'preposition']].drop_duplicates()

    encoder = list(set(df.noun1) | set(df.noun2))
    regions = list(set(df.rcc))
    labels = list(set(df.preposition))

    data = np.zeros((len(df), len(encoder) + len(regions)), dtype=np.float32)
    target = np.zeros(len(df))

    for nn, (_, row) in enumerate(df.iterrows()):
        data[nn, encoder.index(row['noun1'])] = 1.
        data[nn, encoder.index(row['noun2'])] = 1.
        data[nn, regions.index(row['rcc'])] = 1.
        target[nn] = labels.index(row['preposition'])

    x_train = data
    y_train = target

    # training
    print("Training.")
    clf = RandomForestClassifier()  # probability=True)
    clf.fit(x_train, y_train)

    # testing
    print("Testing.")
    test_filename = os.path.join(params.dataset_path, 'test.csv')
    df_test = pd.read_csv(test_filename)
    df_test.fillna('', inplace=True)
    data_test = np.zeros((len(df_test), len(encoder) + len(regions)), dtype=np.float32)
    target_test = np.zeros(len(df_test))

    for nn, (_, row) in enumerate(df_test.iterrows()):
        data_test[nn, encoder.index(row['noun1'])] = 1.
        data_test[nn, encoder.index(row['noun2'])] = 1.
        data_test[nn, regions.index(row['rcc'])] = 1.
        target_test[nn] = labels.index(row['preposition'])

    x_test = data_test 
    y_test = target_test
    y_pred = clf.predict(x_test)

    print("Accuracy score {:.4f}".format(accuracy_score(y_test, y_pred)))
    print("Precision score {:.4f}".format(precision_score(y_test, y_pred, average='macro')))
    print("Recall score {:.4f}".format(recall_score(y_test, y_pred, average='macro')))

    df_test['predicted'] = [labels[int(prediction)] if df_test.iloc[nn].rcc != '' else 'unknow' for nn, prediction in enumerate(y_pred)]
    fname = os.path.join(params.dataset_path, 'predicted_relation.csv')
    df_test.to_csv(fname, index=False)
    
    joblib.dump(clf, os.path.join(params.dataset_path, 'rf.model'))
    with open(os.path.join(params.dataset_path, 'encoder.json'), 'w') as fp:
        json.dump({'encoder': encoder, 'regions': regions, 'labels': labels}, fp)

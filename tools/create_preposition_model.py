# coding: utf-8
import os
import sys
path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, path)
from irrcc.dataset import Dataset
from irrcc.query_annotation import QueryAnnotation


def extract_features()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--data_path', action="store", default='data/sun09_groundTruth.mat')
    # parser.add_argument('-i', '--image_path', action="store", default='images')
    # parser.add_argument('-o', '--output_path', action="store", default='data')
    params = parser.parse_args()

    qa = QueryAnnotation(params.data_path)
    df = pd.DataFrame.from_records(qa.features(), columns=['images', 'noun1', 'preposition', 'noun2'])

    df['rcc'] = df.apply(lambda x: compute_relation(x['images'], x['noun1'], x['noun2']), axis=1)
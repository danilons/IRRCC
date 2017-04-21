# coding: utf-8
import os
import argparse
import click
import time
import json
import re
import numpy as np
import pandas as pd
import sys
import subprocess 
path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, path)
from irrcc.dataset import Dataset
from irrcc.query_annotation import QueryAnnotation


class KnowledgeBase:
    def __init__(self, df, ltb_runner, eprover, batch_config):
        self.dset = Dataset('data', 'test', 'images/')
        self.ltb_runner = ltb_runner
        self.eprover = eprover
        self.batch_config = batch_config
        self.df = df[df.predicted != 'unknow']  # there's a typo
        self.images = list(set(df.images))
        self.regex = re.compile(b'\[\[(\w+)')

    def ontology_by_image(self, image):
        imgname = image
        imindex = self.images.index(image)
        frame = self.df[self.df['images'] == imgname]
        for _ in xrange(1):
            yield "(instance {} Image)".format(imgname).replace(".jpg", "")

        objects = set(frame.noun1) | set(frame.noun2)
        for obj in objects:
            objname = "{}{}".format(obj, imindex)
            yield "(instance {} {})".format(objname, obj.title())

        for obj in objects:
            objname = "{}{}".format(obj, imindex)
            yield "(contains {} {})".format(imgname.replace(".jpg", ""),
                                            objname)

        for _, row in frame.iterrows():
            obj1 = "{}{}".format(row.noun1, imindex)
            obj2 = "{}{}".format(row.noun2, imindex)
            yield "(orientation {} {} {})".format(obj1,
                                                  obj2,
                                                  row['predicted'].title()).replace("_", "")

    def tptp_by_image(self, image, position=0):
        for axiom in self.ontology_by_image(image):
            if axiom.startswith("(instance"):
                _, instance, classname = axiom.replace(")", "").split()
                yield "fof(kb_IRRC_{},axiom,(( s__instance(s__{}__m,s__{}) ))).".format(position,
                                                                                        instance,
                                                                                        classname)

            if axiom.startswith("(contains"):
                _, owner, element = axiom.replace(")", "").split()
                yield "fof(kb_IRRC_{},axiom,(( s__contains(s__{}__m, s__{}__m) ))).".format(position,
                                                                                            owner,
                                                                                            element)

            if axiom.startswith("(orientation"):
                _, obj1, obj2, relation = axiom.replace(")", "").split()
                yield "fof(kb_IRRC_{},axiom,(( s__orientation(s__{}__m,s__{}__m,s__{}) ))).".format(position, obj1, obj2, relation)
            position = position + 1

    def tptp_query(self, image, noun1, noun2, preposition):
        prep = preposition.title().replace("_", "")
        return """fof(conj1,conjecture, ( (? [V__X1,V__X2,V__X3] :
                    (s__instance(V__X1,s__Image) &
                     s__instance(V__X2,s__{noun1}) &
                     s__instance(V__X3,s__{noun2}) &
                     s__contains(V__X1,V__X2) &
                     s__contains(V__X1,V__X3) &
                     s__orientation(V__X2,V__X3,s__{prep}) &
                     s__instance(s__{img}__m, s__Image))) )).
               """.format(noun1=noun1.title(), noun2=noun2.title(), prep=prep, img=image.replace(".jpg", ""))

    def prover(self, image, query):
        try:
            noun1, preposition, noun2 = query.split('-')
        except AttributeError:
            return image, None, []

        objects = {self.dset.segmentation.classes[k] for k in np.unique(np.array(self.dset.segmentation.objects[image]))}
        if noun1 not in objects or noun2 not in objects:
            return image, None, []

        tptp_query = self.tptp_query(image, noun1, noun2, preposition)
        with open('IRRC.tptp', 'w') as fp:
            for axiom in self.tptp_by_image(image):
                fp.write(axiom + "\n")

        with open('Problems.p', 'w') as fp:
            fp.write(tptp_query)

        with open("Answers.p", "w") as fp:
            fp.write("")

        cmd = '{} {} {}'.format(self.ltb_runner, self.batch_config, self.eprover)
        with open(os.devnull, 'wb') as devnull:
            _ = subprocess.call([self.ltb_runner, self.batch_config, self.eprover], 
                                 stdout=devnull, stderr=subprocess.STDOUT)

        response = []
        with open("Answers.p", "r") as fp:
            for line in fp.readlines():
                response.append(line.strip())

        m = self.regex.findall("\n".join(response))
        if m:
            return image, m[0], response
        return image, None, response

    def runquery(self, query):
        for image in self.images:
            image, imname, _ = self.prover(image=image, query=query)
            yield image, imname


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def evaluate(queries, ground_truth, kb):
    mean_average_precision = []
    valid_queries = [v for v in queries.values() if v is not None]
    for nn, query in enumerate(ground_truth.db):
        print("Processed {}/{}".format(nn, len(ground_truth.db)))
        equivalent = queries.get(query)
        if equivalent:
            retrieved = [] 
            with click.progressbar(length=len(kb.images), show_pos=True, show_percent=True) as bar:
                for returned, found in kb.runquery(equivalent):
                    if found:
                        retrieved.append(returned)
                    bar.update(1)

            score = apk(ground_truth.db[query], retrieved)
            print(u"query {} returned {} images, ground-truth has {}. Score {:.4f}".format(query, len(retrieved), len(ground_truth.db[query]), score))
            mean_average_precision.append(score)
        

    return mean_average_precision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/segmentation/predicted_relation.csv')
    parser.add_argument('-q', '--queries_path', action="store", default='data/segmentation/query_equivalence.csv')
    parser.add_argument('-t', '--test_anno', action="store", default='data/segmentation/test_anno/')
    parser.add_argument('--ltb_runner', action="store", default='data/segmentation/ltb_runner')
    parser.add_argument('--eprover', action="store", default='data/segmentation/eprover')
    parser.add_argument('--batch_config', action="store", default='EBatchConfig.txt')
    params = parser.parse_args()

    df = pd.read_csv(params.dataset_path)
    kb = KnowledgeBase(df, ltb_runner=params.ltb_runner, eprover=params.eprover, batch_config=params.batch_config)

    queries = pd.read_csv(params.queries_path)
    queries = dict(zip(queries['Original'], queries['Equivalent']))

    location = os.path.join(params.test_anno)
    qa = QueryAnnotation(location)
    mean_average_precision = evaluate(queries, qa, kb)
    with open(os.path.join('data/segmentation/map_prover.json'), 'w') as fp:
        json.dump(mean_average_precision, fp)
    print("mAP {:.4f}".format(np.mean(mean_average_precision)))

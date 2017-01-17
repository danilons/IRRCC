# coding: utf-8
import os
from glob import glob


class QueryAnnotation:
    def __init__(self, path):
        self.db = {}
        self.imgs = {}
        for folder in glob(os.path.join(path, '*')):
            for fname in glob(os.path.join(folder, '*.txt')):
                key = os.path.basename(folder) 
                with open(fname, 'r') as fp:
                    self.db[key] = [line.strip() for line in fp.readlines() if line.strip().endswith('.jpg')]
        
        names1, preposition, names2 = zip(*(query.split('-') for query in self.db))
        self.names = list(set(names1) | set(names2))
        self.preposition = list(set(preposition))

        for k, imlist in self.db.iteritems():
            for imname in imlist:
                self.imgs.setdefault(imname, []).append(k)
        
    def __getitem__(self, imname):
        return self.imgs.get(imname, [])
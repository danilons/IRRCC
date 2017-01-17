# coding: utf-8
import numpy as np
from scipy.io import loadmat


class StructuredQuery:
    def __init__(self, fname):
        self.db = loadmat(fname, struct_as_record=True, chars_as_strings=True, squeeze_me=True)
        valid_queries = [q for q in self.query if len(np.nonzero(q['rank'])[0].tolist()) > 30]
        
        self.queries = {}
        self.queries['a'] = [dict(zip(q.dtype.names, q)) for q in valid_queries 
                             if isinstance(q['unary'], np.ndarray) and q['unary'].size == 0 and q['binary'].size == 3]
        self.queries['b'] = [dict(zip(q.dtype.names, q)) for q in self.query if q['unary'] != 0 and q['binary'].size == 3]
        self.queries['c'] = [dict(zip(q.dtype.names, q)) for q in self.query 
                             if isinstance(q['unary'], np.ndarray) and q['unary'].size == 0 and q['binary'].size == 6]
        self.queries['d'] = [dict(zip(q.dtype.names, q)) for q in self.query if q['unary'] != 0 and q['binary'].size == 6]
        self.queries['e'] = [dict(zip(q.dtype.names, q)) for q in self.query 
                             if isinstance(q['unary'], np.ndarray) and q['unary'].size == 0 and q['binary'].size == 9]
        
    @property
    def names(self):
        return self.db['names']
    
    @property
    def relations(self):
        return self.db['relations']
    
    @property
    def query(self):
        return self.db['Query']
    
    @property
    def query_types(self):
        return self.queries.keys()
    
    def __getitem__(self, query_type):
        queries = []
        for query in self.queries.get(query_type, []):
            unary = None
            if query['unary']:
                unary = self.names[query['unary'] - 1]
            try:
                name1 = self.names[query['binary'][:, 0].squeeze() - 1]
                prepo = self.relations[query['binary'][:, 2].squeeze() - 1]
                name2 = self.names[query['binary'][:, 1].squeeze() - 1]
            except IndexError:
                name1 = self.names[query['binary'][0].squeeze() - 1]
                prepo = self.relations[query['binary'][2].squeeze() - 1]
                name2 = self.names[query['binary'][1].squeeze() - 1]
            
            query_name = np.vstack((name1, prepo, name2)).reshape(3, -1).T
            if unary:
                name = unary + ", " + " & ".join(["-".join(q) for q in query_name])
            else:
                name = " & ".join(["-".join(q) for q in query_name])
            queries.append({'binary': query['binary'] - 1, 
                            'unary': query['unary'] - 1, 
                            'name': name,
                            'rank': query['rank'].astype(bool)
                           })
        return queries
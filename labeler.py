# coding: utf-8

class ImagesLabeler:
    def __init__(self, fname, max_items=32):
        self.max_items = max_items
        self.classes = {}
        with open(fname, 'r') as fp:
            for nn, line in enumerate(fp.readlines()):
                if nn == self.max_items:
                    break
                _, classname, _ = line.strip().split()
                self.classes[classname] = nn
        self.background = '__background__'
        self.classes[self.background] = 32
        
        self.colors = [(64, 128, 64),
                       (192, 0, 128),
                       (0, 128, 192),
                       (0, 128, 64),
                       (128, 0, 0),
                       (64, 0, 128),
                       (64, 0, 192),
                       (192, 128, 64),
                       (192, 192, 128),
                       (64, 64, 128),
                       (128, 0, 192),
                       (192, 0, 64),
                       (128, 128, 64),
                       (192, 0, 192),
                       (128, 64, 64),
                       (64, 192, 128),
                       (64, 64, 0),
                       (128, 64, 128),
                       (128, 128, 192),
                       (0, 0, 192),
                       (192, 128, 128),
                       (128, 128, 128),
                       (64, 128, 192),
                       (0, 0, 64),
                       (0, 64, 64),
                       (192, 64, 128),
                       (128, 128, 0),
                       (192, 128, 192),
                       (64, 0, 64),
                       (192, 192, 0),
                       (0, 192, 192),
                       (64, 192, 0),
                       (0, 0, 0)]
        
    def get_color(self, classname):
    	try:
        	return self.colors[self.classes[classname]]
    	except KeyError:
    		return self.colors[self.classes[self.background]]
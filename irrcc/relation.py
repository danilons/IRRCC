# coding: utf-8

import numpy as np
import cv2
import collections


class Relation(object):

    _longname = {'DC': 'Disconnected',
                 'EC': 'Externally connected',
                 'PO': 'Partially Overlapping',
                 'EQ': 'Equal',
                 'TPP': 'Tangential proper part',
                 'TPPi': 'Tangential proper part inverse',
                 'NTPP': 'Non-Tangential proper part',
                 'NTPPi': 'Non-Tangential proper part inverse'}

    def __init__(self, scope):
        self.scope = scope
        self.relation = collections.OrderedDict({'rcc8': scope, 'rcc5': scope})

    def __repr__(self):
        return "RCC relation {}: {}".format(self.scope, self._longname[self.scope])

    def __len__(self):
        return 2

    def __getitem__(self, item):
        return self.relation[item]

    @classmethod
    def get_name(cls, relation):
        return cls._longname[relation]


class RCC(object):

    def compute(self, shape, object1, object2, kernel=None):

        imx = np.zeros(shape, dtype=np.float32)
        cv2.fillPoly(imx, [object1], 1)

        imy = np.zeros(shape, dtype=np.float32)
        cv2.fillPoly(imy, [object2], 2)

        return self.detect(imx, imy, kernel)

    def detect(self, imx, imy, kernel=None):

        hx, hy, ha = self._intersection(imx, imy)

        # define relations
        if hx == 0 and hy == 0 and ha != 0:
            return Relation(scope='EQ')

        if hx != 0 and hy != 0 and ha != 0:
            return Relation(scope='PO')

        # structuring element
        kernel = kernel or np.ones((11, 11), dtype=np.uint8)

        if hx == 0 and hy != 0 and ha != 0:
            imd = cv2.dilate(imx, kernel)
            dilation, _, _ = self._intersection(imd, imy)
            if dilation:
                return Relation(scope='NTPP')
            else:
                return Relation(scope='TPP')

        if hx != 0 and hy == 0 and ha != 0:
            imd = cv2.dilate(imy, kernel)
            dilation, _, _ = self._intersection(imx, imd)
            if dilation:
                return Relation(scope='NTPPi')
            else:
                return Relation(scope='TPPi')

        if hx != 0 and hy != 0 and ha == 0:
            imd = cv2.dilate(imx, kernel)
            _, _, dilation = self._intersection(imd, imy)
            if dilation:
                return Relation(scope='DC')
            else:
                Relation(scope='EC')

        return Relation(scope='DC')

    def _intersection(self, ix, iy):
        """
        Compute region intersection from two binary images

        :param ix: image x
        :param iy: image y
        :return: area of regions (x, y, intersection)
        """
        im = ix + iy  # cv2.bitwise_or(ix, iy)
        # freq = cytoolz.frequencies(im.ravel())
        # hx = freq.get(1, 0)  # x alone
        # hy = freq.get(2, 0)  # y alone
        # ha = freq.get(3, 0)  # x & y

        freq = np.bincount(im.ravel().astype(np.int64))
        hx = freq[1] if len(freq) > 1 else 0
        hy = freq[2] if len(freq) > 2 else 0
        ha = freq[3] if len(freq) > 3 else 0

        return hx, hy, ha

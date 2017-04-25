# coding: utf-8
import os
import sys
path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, path)
import argparse
import signal
import pandas as pd
import h5py
import cv2
import numpy as np
import click
import traceback
from functools import partial
from multiprocessing import Pool

palette = np.array(\
[[ 0, 0, 0],
[128, 0, 0],
[ 0, 128, 0],
[128, 128, 0],
[ 0, 0, 128],
[128, 0, 128],
[ 0, 128, 128],
[128, 128, 128],
[64, 0, 0],
[192, 0, 0],
[64, 128, 0],
[192, 128, 0],
[64, 0, 128],
[192, 0, 128],
[64, 128, 128],
[192, 128, 128],
[ 0,64, 0],
[128,64, 0],
[ 0, 192, 0],
[128, 192, 0],
[ 0,64, 128],
[128,64, 128],
[ 0, 192, 128],
[128, 192, 128],
[64,64, 0],
[192,64, 0],
[64, 192, 0],
[192, 192, 0],
[64,64, 128],
[192,64, 128],
[64, 192, 128],
[192, 192, 128],
[ 0, 0,64],
[128, 0,64],
[ 0, 128,64],
[128, 128,64],
[ 0, 0, 192],
[128, 0, 192],
[ 0, 128, 192],
[128, 128, 192],
[64, 0,64],
[192, 0,64],
[64, 128,64],
[192, 128,64],
[64, 0, 192],
[192, 0, 192],
[64, 128, 192],
[192, 128, 192],
[ 0,64,64],
[128,64,64],
[ 0, 192,64],
[128, 192,64],
[ 0,64, 192],
[128,64, 192],
[ 0, 192, 192],
[128, 192, 192],
[64,64,64],
[192,64,64],
[64, 192,64],
[192, 192,64],
[64,64, 192],
[192,64, 192],
[64, 192, 192],
[192, 192, 192],
[32, 0, 0],
[160, 0, 0],
[32, 128, 0],
[160, 128, 0],
[32, 0, 128],
[160, 0, 128],
[32, 128, 128],
[160, 128, 128],
[96, 0, 0],
[224, 0, 0],
[96, 128, 0],
[224,  128, 0],
[96, 0, 128],
[224, 0, 128],
[96, 128, 128],
[224, 128, 128],
[32,64, 0],
[160,64, 0],
[32, 192, 0],
[160, 192, 0],
[32,64, 128],
[160,64, 128],
[32, 192, 128],
[160, 192, 128],
[96,64, 0],
[224,64, 0],
[96,  192, 0],
[224,  192, 0],
[96,64, 128],
[224,64, 128],
[96, 192, 128],
[224, 192, 128],
[32, 0,64],
[160, 0,64],
[32, 128,64],
[160, 128,64],
[32, 0, 192],
[160, 0, 192],
[32,  128, 192],
[160, 128, 192],
[96, 0,64],
[224, 0, 64]])


class ColorPalette:
    def __init__(self, classnames, palette):
        self.palette = palette
        self.names = ['__background__'] + sorted(set(classnames.values()))
        self.classenames = classnames

    def __getitem__(self, name):
        classname = self.classenames[name]
        index = self.names.index(classname)
        return self.palette[index, :]

    def class_id(self, name):
        classname = self.classenames.get(name, name)
        return self.names.index(classname)

    def get_name(self, class_id):
        return self.names[class_id]

    def get_original_names(self, name):
        names = []
        for k, v in self.classenames.iteritems():
            if name == v:
                names.append(k)
        return names

class Dataset(object):
    def __init__(self, path, suffix='train', image_path='images'):
        self.coordinates = h5py.File(os.path.join(path, 'dataset_{}.hdf5'.format(suffix)))
        self.image_path = image_path

    @property
    def images(self):
        return self.coordinates.keys()

    def ground_truth(self, image):
        gold_standard = self.coordinates[image]
        contour = {}
        for classname in gold_standard:
            bbox = gold_standard.get(classname)
            contour[classname] = np.vstack((bbox['x'], bbox['y'])).T
        return contour

    def get_im_array(self, image, rgb=False):
        if rgb:
            return cv2.imread(os.path.join(self.image_path, image))[:, :, (2, 1, 0)]
        return cv2.imread(os.path.join(self.image_path, image))

    def get_image_with_objects(self, image, obj_id=None, **kwargs):
        img = self.get_im_array(image, **kwargs)
        self.detector.get_image_with_objects(img, image, obj_id, **kwargs)
        return img


def create_labels(imname, ground_truth, color_palette, output_folder, image_folder, suffix):
    img = cv2.imread(os.path.join(image_folder, imname))
    if img is None:
        return False

    w, h = (384, 384)
    w1, h1 = img.shape[:2]
    fy = w / float(w1)
    fx = h / float(h1)
    scale = np.array([fx, fy])

    imagename = os.path.join(output_folder, 'images', suffix, imname)
    if not os.path.exists(imagename):
        image = cv2.resize(img, (0, 0), fx=fx, fy=fy)
        cv2.imwrite(imagename, image)

    labelname = os.path.join(output_folder, 'labels', suffix, imname.replace('.jpg', '.png'))
    if not os.path.exists(labelname):
        label = np.zeros((w, h, 3), dtype=np.uint8)
        for (classname, contour) in ground_truth.items():
            color = color_palette[classname]
            cnt = contour * scale
            cv2.drawContours(label, [cnt.astype(np.int32)], -1, color, -1)

        cv2.imwrite(labelname, label)

    labelname = os.path.join(output_folder, 'mark', suffix, imname.replace('.jpg', '.png'))
    if not os.path.exists(labelname):
        label = np.zeros((w, h), dtype=np.uint8)
        for (classname, contour) in ground_truth.items():
            color = color_palette.class_id(classname)
            cnt = contour * scale
            cv2.drawContours(label, [cnt.astype(np.int32)], -1, color, -1)

        cv2.imwrite(labelname, label)
    return True

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/')
    parser.add_argument('-i', '--image_path', action="store", default='images')
    parser.add_argument('-o', '--output_path', action="store", default='experiments/structured_queries')
    parser.add_argument('-a', '--alias_file', action="store", default='experiments/structured_queries/name_conversion.csv')
    params = parser.parse_args()

    alias = pd.read_csv(params.alias_file).set_index('Class')['Name'].to_dict()
    color_palette = ColorPalette(alias, palette)

    processes = []
    pool = Pool(processes=8, initializer=init_worker)

    for suffix in ['train', 'test']:
        dataset = Dataset(path=params.dataset_path, suffix=suffix, image_path=params.image_path)
        output_folder = params.output_path
        # create partial
        _task = partial(create_labels, color_palette=color_palette,
                        output_folder=output_folder, image_folder=params.image_path, suffix=suffix)

        print("Scheduling for: {}".format(suffix))
        with click.progressbar(length=len(dataset.images), show_pos=True, show_percent=True) as bar:
            for imname in dataset.images:
                ground_truth = {k: v for k, v in dataset.ground_truth(imname).items() if k in alias}
                processes.append(pool.apply_async(func=_task, args=(imname, ground_truth)))
                bar.update(1)

    print("Number of images to create: {}".format(len(processes)))
    with click.progressbar(length=len(processes), show_pos=True, show_percent=True) as bar:
        try:
            for process in processes:
                try:
                    _ = process.get()
                except KeyboardInterrupt:
                    raise
                except:
                    traceback.print_exc()
                bar.update(1)
        except KeyboardInterrupt:
            print "Caught KeyboardInterrupt, terminating workers"
            pool.terminate()
            pool.join()
            sys.exit(0)



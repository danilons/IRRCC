# coding: utf-8
import os
import click
import glob
import cv2
import argparse
import caffe
import h5py
from fast_rcnn.test import im_detect
from fast_rcnn.config import cfg
cfg.TEST.HAS_RPN = True  # Use RPN for proposals


def detect_objects(net, input_file, image_path, output_file):
    hdf5 = h5py.File(input_file)
    objects_hdf5 = h5py.File(output_file, 'w')
    with click.progressbar(length=len(hdf5), show_pos=True, show_percent=True) as bar:
        for imname in hdf5:
            group = objects_hdf5.create_group(imname)

            im = cv2.imread(os.path.join(image_path, imname))
            if im is None:
                print("Unable to read image: {}".format(os.path.join(image_path, imname)))
                continue

            scores, boxes = im_detect(net, im)
            group['scores'] = scores
            group['boxes'] = boxes

            bar.update(1)

    objects_hdf5.close()
    print("Saved objects to: {}".format(output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/')
    parser.add_argument('-p', '--prototxt', action="store", default='data/test.prototxt')
    parser.add_argument('-w', '--weights', action="store", default='data/rcnn.caffemodel')
    parser.add_argument('-i', '--image_path', action="store", default='Images/static_sun09_database')
    parser.add_argument('--gpu', dest='gpu', action="store_true", default=True)
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    parser.set_defaults(feature=True)
    params = parser.parse_args()

    if params.gpu:
        caffe.set_mode_gpu()

    net = caffe.Net(params.prototxt, params.weights, caffe.TEST)

    files = glob.glob(os.path.join(params.dataset_path, 'dataset_*.hdf5'))
    for fname in files:
        print("Start processing file: {}".format(fname))
        output_file = fname.replace('dataset_', 'objects_')
        detect_objects(net, fname, params.image_path, output_file)

    print("Done.")

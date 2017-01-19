# coding: utf-8
import os
import glob
import click
import cv2
import numpy as np
import argparse
from dataset import Dataset
from labeler import ImagesLabeler


def segmentation_labels(dset, labeler, mode, output_folder):
    with click.progressbar(length=len(dset.images), show_pos=True, show_percent=True) as bar:
        for imname in dset.images:
            image = dset.get_im_array(imname)
            if image is None:
                print("Not found image {}".format(imname))
                bar.update(1)
                continue

            lbl = np.zeros(image.shape, dtype=np.uint8)
            gs = dset.ground_truth(imname)
            for classname, contour in gs.iteritems():
                color = labeler.get_color(classname)
                cv2.drawContours(lbl, [contour.astype(np.int32)], -1, color, -3)
            cv2.imwrite(os.path.join('original', mode, imname), image)
            cv2.imwrite(os.path.join(output_folder, mode, imname.replace('.jpg', '.png')), lbl)
            bar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/')
    parser.add_argument('-o', '--objects_file', action="store", default='data/objects.txt')
    parser.add_argument('-p', '--output_path', action="store", default='segmentation')
    parser.add_argument('-i', '--image_path', action="store", default='images')
    params = parser.parse_args()
    
    labeler = ImagesLabeler(fname=params.objects_file)

    files = glob.glob(os.path.join(params.dataset_path, 'dataset_*.hdf5'))
    for fname in files:
        print("Start processing file: {}".format(fname))

        basename = os.path.basename(fname)
        filename, _ = os.path.splitext(basename)
        mode = filename.replace('dataset_', '')
        print("Processing mode: {}".format(mode))
        dset = Dataset(params.dataset_path, mode, params.image_path)

        segmentation_labels(dset, labeler, mode, params.output_path)

    print("Done.")


# coding: utf-8
import os
import json
import click
import argparse
import scipy.io as sio
import h5py
import numpy as np


def save_image_list(dataset, image_path, output_path, mode):
    basename = os.path.join(output_path, 'dataset_{}.hdf5'.format(mode))
    if os.path.exists(basename):
        os.remove(basename)

    hdf5 = h5py.File(basename, 'w')
    imagenames = []
    with click.progressbar(length=len(dataset), show_pos=True, show_percent=True) as bar:
        for row_number, annotation in enumerate(dataset['annotation']):
            filename = unicode(annotation['filename'])
            imagenames.append(filename)
            try:
                hdf5_group = hdf5.create_group(filename)
            except ValueError:
                hdf5_group = hdf5.create_group("{}_{}".format(filename, row_number))

            imagefile = os.path.join(image_path, filename)
            if not os.path.exists(imagefile):
                print("File not found: {}".format(imagefile))

            objects = annotation['object'].item()
            if annotation['object'].item().size == 1:
                objects = [annotation['object'].item()]

            for nn, obj in enumerate(objects):
                try:
                    name = unicode(obj['name'])
                    hdf5_group.create_group(name)
                except ValueError:
                    name = u"{}{}".format(obj['name'], nn)
                    hdf5_group.create_group(name)

                try:
                    hdf5_group[name].create_dataset('x', data=obj['polygon']['x'].item(), compression="gzip")
                    hdf5_group[name].create_dataset('y', data=obj['polygon']['y'].item(), compression="gzip")
                except IndexError:
                    hdf5_group[name].create_dataset('x', data=obj['polygon'].item()['x'].item(), compression="gzip")
                    hdf5_group[name].create_dataset('y', data=obj['polygon'].item()['y'].item(), compression="gzip")


            bar.update(1)
    
    with open(os.path.join(output_path, 'imagenames_{}.json'.format(mode)), 'w') as fp:
        json.dump(imagenames, fp)

    hdf5.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-g', '--ground_truth', action="store", default='data/sun09_groundTruth.mat')
    parser.add_argument('-i', '--image_path', action="store", default='images')
    parser.add_argument('-o', '--output_path', action="store", default='data')
    params = parser.parse_args()

    print("Reading data")
    dset = sio.loadmat(params.ground_truth,
                       struct_as_record=True,
                       chars_as_strings=True,
                       squeeze_me=True)

    trainset = {'train': 'Dtraining', 'test': 'Dtest'}
    for mode, dset_key in trainset.iteritems():
        print("{}:-----------------------------".format(mode.title()))
        save_image_list(dset[dset_key],
                        params.image_path,
                        params.output_path,
                        mode)
        print("Done.")

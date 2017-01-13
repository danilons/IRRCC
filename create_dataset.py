# coding: utf-8
import os
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
    with click.progressbar(length=len(dataset), show_pos=True, show_percent=True) as bar:
        for row_number, annotation in enumerate(dataset['annotation']):
            filename = unicode(annotation['filename'])
            imagefile = os.path.join(image_path, filename)
            if not os.path.exists(imagefile):
                print("File not found: {}".format(imagefile))

            if filename not in hdf5:
                hdf5_group = hdf5.create_group(filename)
                if annotation['object'].item().size == 1:
                    objects = [annotation['object'].item()]
                else:
                    objects = annotation['object'].item()

                for obj in objects:
                    name = unicode(obj['name'])
                    if name not in hdf5_group:
                        hdf5_group.create_group(name)

                    try:
                        polygon = np.vstack((obj['polygon']['x'].item(),
                                             obj['polygon']['y'].item())).T
                    except:
                        polygon = np.vstack((obj['polygon'].item()['x'].item(),
                                             obj['polygon'].item()['y'].item())).T

                    hdf5_group[name].create_dataset(unicode(obj['id']), data=polygon, compression="gzip")

            bar.update(1)

    hdf5.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-g', '--ground_truth', action="store", default='sun09_groundTruth.mat')
    parser.add_argument('-i', '--image_path', action="store", default='Images/static_sun09_database')
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

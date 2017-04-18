# coding: utf-8
import os
import sys
import glob
import click
import argparse
import json
import h5py
import numpy
path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, path)
from irrcc.dataset import Dataset


def store_topology(dset, output_file):
    filename = output_file.replace('#', 'segment')
    if os.path.exists(filename):
        print("Removing file: {}".format(filename))
        os.remove(filename)

    hdf5 = h5py.File(filename, 'w')
    with click.progressbar(length=len(dset.images), show_pos=True, show_percent=True) as bar:
        for image in dset.images:
            if image in hdf5:
                continue
                
            try: 
                relations = dset.segmentation.topology_relation((384, 384), image)
            except KeyError, ValueError:
                bar.update(1)
                continue

            hdf5_group = hdf5.create_group(image)
            for topology in relations:
                objects = '-'.join(topology['objects']).strip()
                try:
                    hdf5_group.create_group(objects)
                    hdf5_group[objects].create_dataset('contours1', data=topology['contours'][0], compression="gzip")
                    hdf5_group[objects].create_dataset('contours2', data=topology['contours'][1], compression="gzip")
                    hdf5_group[objects]['relation'] = numpy.string_(topology['relation'])
                except ValueError:
                    print("Error when processing image {} and {}".format(image, objects))
                except RuntimeError:
                    print("Runtime error when processing image {} and {}".format(image, objects))
                    continue
                except IOError:
                    print("IOError error when processing image {} and {}".format(image, objects))
            bar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/')
    parser.add_argument('-i', '--image_path', action="store", default='images')
    params = parser.parse_args()

    files = glob.glob(os.path.join(params.dataset_path, 'dataset_*.hdf5'))
    for fname in files:
        print("Start processing file: {}".format(fname))
        
        basename = os.path.basename(fname)
        filename, _ = os.path.splitext(basename)
        mode = filename.replace('dataset_', '')
        print("Processing mode: {}".format(mode))

        dset = Dataset(params.dataset_path, mode, params.image_path)
        output_file = fname.replace('dataset_', 'topology_#_')
        store_topology(dset, output_file)

    print("Done.")

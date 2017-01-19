# coding: utf-8
import os
import glob
import click
import argparse
import json
from irrcc.dataset import Dataset


def store_topology(dset, output_file):
    relations1 = {}
    relations2 = {}
    with click.progressbar(length=len(dset.images), show_pos=True, show_percent=True) as bar:
        for image in dset.images:
            try: 
                topology1 = dset.objects.topology_relation(image)
                topology2 = dset.segmentation.topology_relation(image)
            except KeyError:
                bar.update(1)
                continue
            relations[image] = []
            for box1, box2, relation in topology1:
                relations1[image].append({'box1': box1.tolist(), 
                                          'box2': box2.tolist(), 'rcc': relation})

            for cnt1, cnt2, relation in topology2:
                relations1[image].append({'box1': cnt1.tolist(), 
                                          'box2': cnt2.tolist(), 'rcc': relation})
            bar.update(1)

    with open(output_file.replace('#', 'boxes'), 'w') as fp:
        json.dump(relations1, fp)

    with open(output_file.replace('#', 'segment'), 'w') as fp:
        json.dump(relations2, fp)

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
        output_file = fname.replace('dataset_', 'topology_#').replace('hdf5', 'json')
        store_topology(dset, output_file)

    print("Done.")
# coding: utf-8
import os
import click
import glob
import cv2
import argparse
import caffe
import h5py


def segment_objects(net, input_file, image_path, output_folder):
    hdf5 = h5py.File(input_file)
    net.blobs['data'].reshape(1, 3, 384, 384)
    with click.progressbar(length=len(hdf5), show_pos=True, show_percent=True) as bar:
        for imname in hdf5:
            imout = os.path.join(output_folder, imname.replace('.jpg', '.png'))
            if os.path.exists(imout):
                print("  Skipping: {}".format(imname))
                bar.update(1)
                continue

            im = cv2.imread(os.path.join(image_path, imname))
            if im is None:
                print("Unable to read image: {}".format(os.path.join(image_path, imname)))
                bar.update(1)
                continue

            # w, h, c = im.shape
            # # if w >= 1503  and h >= 1917:  # avoid overflow
            # if h >= 1649:  # avoid overflow
            #     print("  Wrong size: {}: ({})".format(imname, im.shape))
            #     bar.update(1)
            #     continue

            print("  Processing: {}".format(imname))

            # net.blobs['data'].reshape(1, c, w, h)
            transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
            transformer.set_transpose('data', (2, 0, 1))
            transformer.set_raw_scale('data', 255)
            transformer.set_channel_swap('data', (2, 1, 0))

            transformed_image = transformer.preprocess('data', im)
            net.blobs['data'].data[...] = transformed_image

            output = net.forward()
            segmentation = output['fc_final_up'][0].argmax(axis=0)
            imout = os.path.join(output_folder, imname.replace('.jpg', '.png'))
            cv2.imwrite(imout, segmentation)
            bar.update(1)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRRCC program')
    parser.add_argument('-d', '--dataset_path', action="store", default='data/')
    parser.add_argument('-p', '--prototxt', action="store", default='data/sceneparsing/deploy_DilatedNet.prototxt')
    parser.add_argument('-w', '--weights', action="store", default='data/sceneparsing/DilatedNet.caffemodel')
    parser.add_argument('-i', '--image_path', action="store", default='images')
    parser.add_argument('-o', '--output_folder', action="store", default='segmentation2')
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
        segment_objects(net, fname, params.image_path, params.output_folder)

    print("Done.")

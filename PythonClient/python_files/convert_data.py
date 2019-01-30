
import dataset_loader
import models
import numpy as np
import sys
import argparse
import time
import os

def main(args):
    t = time.time()
    base_path_input = '/run/media/vlad/326c1b61-b264-486c-b0cb-4258027fb67a/datasets/'
    base_path_output = '/home/vlad/nvidia/external_datasets/'
    path = os.path.join(base_path_input, args.input)
    camera_dir = args.sensor

    output_path = os.path.join(base_path_output, args.output)
    input_shape = (args.width, args.height, args.channels)

    dataset = dataset_loader.Dataset(path, camera_dir, limit=args.dataset_limit).get_dataset()
    X = dataset[:, 0]

    try:
        os.makedirs(output_path)
    except:
        pass

    if args.batch_size == -1:
        X = dataset_loader.transform_batch(X, input_shape)
        Y = np.stack(dataset[:, 1:]).astype(np.float64)

        np.save(os.path.join(output_path, 'Y'), Y)
        np.save(os.path.join(output_path, 'X'), X)
        print('done')

    else:
        for i in range(0, len(X), args.batch_size):
            indices = range(i, min(len(X), i + args.batch_size))
            X_ = dataset_loader.transform_batch(X[indices], input_shape)
            Y_ = dataset[indices, 1:].astype(np.float64)

            np.save(os.path.join(output_path, 'Y_{}'.format(i)), Y_)
            np.save(os.path.join(output_path, 'X_{}'.format(i)), X_)
            print('finished {}/{}'.format(i, len(X)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
           '-i', '--input',
           metavar='INPUT',
           dest='input',
           default=None,
           required=True,
           help='input dataset in std format')
    argparser.add_argument(
           '-o', '--output',
           metavar='OUTPUT',
           dest='output',
           default=None,
           required=True,
           help='output dataset')
    argparser.add_argument(
           '-c', '--channels',
           metavar='C',
           default=1,
           type=int,
           help='Number of image channels')
    argparser.add_argument(
           '--width',
           metavar='W',
           default=160,
           type=int,
           help='Input image width')
    argparser.add_argument(
           '--height',
           metavar='H',
           default=120,
           type=int,
           help='Input image height')
    argparser.add_argument(
           '-s', '--sensor',
           metavar='SENSOR',
           dest='sensor',
           default=None,
           required=True,
           help='sensor to train against')
    argparser.add_argument(
           '-l', '--dataset_limit',
           metavar='L',
           default=-1,
           type=int,
           help='limit dataset size')
    argparser.add_argument(
           '-b', '--batch_size',
           metavar='B',
           default=-1,
           type=int,
           help='limit dataset file')

    args = argparser.parse_args()
    main(args)

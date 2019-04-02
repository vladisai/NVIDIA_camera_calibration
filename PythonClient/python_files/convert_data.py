
import dataset_loader
import models
import numpy as np
import sys
import argparse
import time
import os
import pandas as pd

def main(args):
    t = time.time()
    path = args.input
    camera_dir = args.sensor

    output_path = args.output
    input_shape = (args.width, args.height, args.channels)

    dataset = dataset_loader.Dataset(path, camera_dir, limit=args.dataset_limit, pairs=args.pairs).get_dataset()

    X = None
    Y = None
    if args.pairs:
        X = dataset[:, [0, 1]]
        Y = np.stack(dataset[:, 2:]).astype(np.float64)
        X = dataset_loader.transform_double(X, input_shape)
    else:
        X = dataset[:, 0]
        Y = np.stack(dataset[:, 1:]).astype(np.float64)
        X = dataset_loader.transform_batch(X, input_shape)

    try:
        os.makedirs(output_path)
    except:
        pass

    if args.split:
        rows_per_partition = len(X) // args.split
        remainder = len(X) % args.split
        index = 0

        for i in range(0, args.split):
            current = index + rows_per_partition
            if remainder > 0:
                current += 1
                remainder -= 1
            np.save(os.path.join(output_path, 'X_{}'.format(i)), X[index:current])
            pdy = pd.DataFrame(data=Y[index:current], 
                    columns=['roll', 'pitch', 'yaw', 'speed', 'steer', 'throttle', 'brake']
                    )
            pdy.to_csv(os.path.join(output_path, 'Y_{}.csv'.format(i)))
            index = current
    else:
        #np.save(os.path.join(output_path, 'Y'), Y)
        np.save(os.path.join(output_path, 'X'), X)
        pdy = pd.DataFrame(data=Y, 
                columns=['roll', 'pitch', 'yaw', 'speed', 'steer', 'throttle', 'brake']
                )
        pdy.to_csv(os.path.join(output_path, 'Y.csv'))

    print('done')

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
           '-p', '--pairs',
           action='store_true',
           default=False,
           help='join images into pairs')
    argparser.add_argument(
            '--split',
            metavar='SPLIT',
            default=None,
            type=int,
            help='split into multiple files'
    )
    args = argparser.parse_args()
    main(args)

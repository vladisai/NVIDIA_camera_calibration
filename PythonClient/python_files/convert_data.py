
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
    path = '/home/vlad/datasets/{}'.format(args.input)
    camera_dir = args.sensor

    output_path = '/home/vlad/datasets/{}'.format(args.output)
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

    np.save(os.path.join(output_path, 'Y'), Y)
    np.save(os.path.join(output_path, 'X'), X)
    #pdx = pd.DataFrame(data=X)
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

    args = argparser.parse_args()
    main(args)

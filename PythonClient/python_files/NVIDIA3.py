#!/usr/bin/env python
# coding: utf-8

import dataset_loader
import models
import numpy as np
import sys
import argparse
import time
import ast

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import normalize

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os


def main(args):
    t = time.time()
    #camera_dir = args.sensor

    #dataset = dataset_loader.Dataset(path, camera_dir, limit=args.dataset_limit).get_dataset()

    #X = np.load(dataset[:, 0]
    #Y = np.stack(dataset[:, 2]).astype(np.float64)
    datasets = ast.literal_eval(args.datasets)
    columns = ast.literal_eval(args.columns)

    X, Y = dataset_loader.loadXY(*datasets[0], columns=columns)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, shuffle=False)
    X_test, X_val, Y_test, Y_val = train_test_split(X, Y, test_size=0.5)

    for dataset in datasets[1:]:
        X2, Y2 = dataset_loader.loadXY(*dataset)
        X_train = np.concatenate([X_train, X2])
        Y_train = np.concatenate([Y_train, Y2])

    X_train, Y_train = sklearn.utils.shuffle(X_train, Y_train)

#    print(len(X))
#    inds = np.abs(Y[:, 0]) < 15
#    X = X[inds]
#    Y = Y[inds, 1]
#    print(len(X))

    input_shape = X[0].shape
    print('input shape is ', input_shape)
    if type(Y[0]) == np.float64:
        output_len = 1
    else:
        output_len = len(Y[0])
    print('output len is', output_len)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    model = None
    models_dict = {}
    models_dict['simple'] = models.ModelSimple
    models_dict['simpleMSE'] = models.ModelSimpleMSE
    models_dict['simple2'] = models.ModelSimple2
    models_dict['simple3'] = models.ModelSimple3
    models_dict['simple_good'] = models.ModelSimple_good
    models_dict['NVIDIA'] = models.ModelNVIDIA

    if args.model in models_dict:
        model = models_dict[args.model](input_shape, output_len)
    else:
        model = models.ModelFromFile(args.model)

    hist = model.fit(X_train, Y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_val, Y_val))
    #model.train(X_train, Y_train, X_val, Y_val, epochs=args.epochs, batch_size=args.batch_size)

    train_mae = 'Train MAE: {}'.format(model.validate(X_train, Y_train, verbose=True))
    validation_mae = 'Validation MAE: {}'.format(model.validate(X_val, Y_val, verbose=True))
    test_mae = 'Test MAE: {}'.format(model.validate(X_test, Y_test, verbose=True))

    dummy = np.zeros(Y_test.shape)
    dummy.fill(np.mean(Y_test))
    dummy_mae = 'Dummy MAE: {}'.format(mean_absolute_error(dummy , Y_test))
    total_time = 'total time {}s'.format(time.time() - t)

    if args.save_to:
        model.save(args.save_to)
        with open('{}_logs'.format(args.save_to), 'w') as f:
            f.write(args.model + '\n')
            f.write(str(sys.argv) + '\n')
            f.write(str(hist.history) + '\n')
            f.write(train_mae + '\n')
            f.write(validation_mae + '\n')
            f.write(test_mae + '\n')
            f.write(dummy_mae + '\n')
            f.write(total_time + '\n')

    print(train_mae)
    print(validation_mae)
    print(test_mae)
    print(dummy_mae)
    print(total_time)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
           '-b', '--batch-size',
           metavar='B',
           default=64,
           type=int,
           help='batch size')
    argparser.add_argument(
           '-e', '--epochs',
           metavar='E',
           default=10,
           type=int,
           help='number of epochs')
    argparser.add_argument(
           '-o', '--save-to',
           metavar='SAVE_NAME',
           dest='save_to',
           default=None,
           help='name for the model to save')
    argparser.add_argument(
           '-m', '--load-model',
           metavar='LOAD_MODEL',
           dest='model',
           default='simple',
           help='name for the model to load')
    argparser.add_argument(
           '-d', '--datasets',
           metavar='DATASET',
           dest='datasets',
           default=None,
           required=True,
           help='datasets list in format (\'name\', percentage). The first dataset will be validated and tested against.')
    argparser.add_argument(
           '-c', '--columns',
           metavar='COLUMNS',
           dest='columns',
           default=None,
           required=True,
           help='Columns to predict in Y in format [\'col1\', \'col2\'].')

    args = argparser.parse_args()
    main(args)

#!/usr/bin/env python
# coding: utf-8

import dataset_loader
import dataset_generator
import models
import numpy as np
import sys
import argparse
import time
import ast

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os

def main(args):
    t = time.time()

    datasets = ast.literal_eval(args.datasets)
    columns = ast.literal_eval(args.columns)

    ds_path = os.path.join(args.datasets_root, datasets[0][0])
    gen_train = dataset_generator.DataGenerator(ds_path, args.batch_size, columns=columns, seek=0, read_length=8, augment=args.augmentations)
    gen_val = dataset_generator.DataGenerator(ds_path, args.batch_size, columns=columns, seek=8, read_length=1)
    gen_test = dataset_generator.DataGenerator(ds_path, args.batch_size, columns=columns, seek=9, read_length=1)

    X, Y = dataset_loader.loadXY(args.datasets_root, *datasets[0], columns=columns, suffix=0)
    input_shape = X[0].shape
    print('input shape is ', input_shape)
    output_len = len(columns)
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
    models_dict['NVIDIA_MAE'] = models.ModelNVIDIA_MAE
    models_dict['NVIDIA_MSE'] = models.ModelNVIDIA_MSE

    if args.model in models_dict:
        model = models_dict[args.model](input_shape, output_len)
    else:
        model = models.ModelFromFile(args.model)

    hist = model.fit_generator(gen_train, epochs=args.epochs, validation_data=gen_val)

    print('fitted')

    #train_mse = 'Train MSE: {}'.format(model.evaluate_generator(gen_train))
    validation_mse = 'Validation: {}'.format(model.evaluate_generator(gen_val))
    test_mse = 'Test: {}'.format(model.evaluate_generator(gen_test))
    total_time = 'Total Time: {}s'.format(time.time() - t)

    if args.save_to:
        model.save(args.save_to)
        with open('{}_logs'.format(args.save_to), 'w') as f:
            f.write(args.model + '\n')
            f.write(str(sys.argv) + '\n')
            f.write(str(hist.history) + '\n')
            f.write(validation_mse + '\n')
            f.write(test_mse + '\n')
            f.write(total_time + '\n')
            commit = os.popen('git log -n 1').read()
            f.write(commit)

    print(validation_mse)
    print(test_mse)
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
    argparser.add_argument(
           '--datasets-root',
           metavar='DATASET',
           dest='datasets_root',
           default='/home/vlad/nvidia/datasets',
           help='Datasets root folder.')
    argparser.add_argument(
           '--augmentations',
           metavar='AUG',
           default=0,
           type=int,
           help='Number of augmentations per image')

    args = argparser.parse_args()
    main(args)

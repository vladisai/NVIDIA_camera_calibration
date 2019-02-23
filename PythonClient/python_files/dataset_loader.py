import os
import configparser
import json
import itertools
import numpy as np
from matplotlib.pyplot import imshow
import PIL
import random
import time
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

config_path = 'configs/000000_config'
measurements_dir = 'measurements'

def loadXY(datasets_root, path, percentage, columns, index=None):
    X_name, Y_name = "X.npy", "Y.csv"
    if index is not None:
        X_name, Y_name = "X_{}.npy".format(index), "Y_{}.csv".format(index)
    X = np.load(os.path.join(datasets_root, path, X_name))
    #Y = np.stack(np.load(os.path.join(datasets_root, path, 'Y.npy')))
    Y = pd.read_csv(os.path.join(datasets_root, path, Y_name), usecols=columns)
    Y = Y.values
    amnt = int(len(X) * percentage)
    X = X[:amnt]
    Y = Y[:amnt]
    return X, Y

def read_config(config_path):
    config = configparser.ConfigParser()
    with open(config_path) as fp:
        config.read_file(itertools.chain(['[global]'], fp), source=config_path)
    d = config['global']
    #return np.float64(d['pitch'])
    return d['roll'], d['pitch'], d['yaw']

def read_image(image_path, shape=(160, 120, 1)):
    w, h, c = shape
    img = load_img(image_path)
    if c == 1:
        img = img.convert('L')
    if img.width != w or img.height != h:
        raise Exception('should be same size')
        #img = img.resize((w, h),PIL.Image.ANTIALIAS)
    x = img_to_array(img)
    x /= 255
    return x

def transform_batch(X, shape):
    w, h, c = shape
    result = np.zeros((len(X), h, w, c))
    t = time.time()
    for i in range(len(X)):
        result[i] = read_image(X[i], shape)
        if i % 1000 == 0:
            print('done {}%'.format(100 * (float(i) / len(X))))
            print('eta {}s'.format((time.time() - t) / (i + 1) * (len(X) - i - 1)))
    return result

def transform_double(X, shape):
    w, h, c = shape
    result = np.zeros((len(X), 2, h, w, c))
    t = time.time()
    for i in range(len(X)):
        result[i, 0] = read_image(X[i, 0], shape)
        result[i, 1] = read_image(X[i, 1], shape)
        if i % 1000 == 0:
            print('done {}%'.format(100 * (float(i) / len(X))))
            print('eta {}s'.format((time.time() - t) / (i + 1) * (len(X) - i - 1)))
    return result

def get_car_info(measurements_path):
    json_file = open(measurements_path)
    json_str = json_file.read()
    json_data = json.loads(json_str)
    speed = 0
    steer = 0
    throttle = 0
    brake = 0
    autopilot_dict = json_data['playerMeasurements']['autopilotControl']
    if 'collisionOther' in json_data['playerMeasurements']:
        raise Exception('dont want data about collisions {}'.format(json_data['playerMeasurements']['collisionOther']))
    if 'throttle' in autopilot_dict:
        throttle = autopilot_dict['throttle']
    if 'brake' in autopilot_dict:
        brake = autopilot_dict['brake']
    if 'steer' in autopilot_dict:
        steer = autopilot_dict['steer']
    if 'forwardSpeed' in json_data['playerMeasurements']:
        speed = json_data['playerMeasurements']['forwardSpeed']
    return np.float64(speed), np.float64(steer), np.float64(throttle), np.float64(brake)

read_image_vec = np.vectorize(read_image)

def get_number_from_image_name(path):
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    return int(name)

def check_is_following(path1, path2):
    return get_number_from_image_name(path1) + 1 == get_number_from_image_name(path2)

class Dataset:
    def __init__(self, path, camera_dir, limit = -1, pairs=False):
        dirs = os.listdir(path)
        random.shuffle(dirs)
        self.dataset = []
        for d in dirs:
            self.dataset.extend(self.add_episode(os.path.join(path, d), camera_dir, pairs))
            if limit > 0 and len(self.dataset) > limit:
                break
        self.dataset = np.stack(np.array(self.dataset))
    
    def get_dataset(self):
        return self.dataset

    def add_episode(self, episode_path, camera_dir, pairs=False):
        result = []
        episode_images_dir = os.path.join(episode_path, camera_dir)
        episode_measurements_dir = os.path.join(episode_path, measurements_dir)

        episode_config_path = os.path.join(episode_path, config_path)
        pitch = read_config(episode_config_path)

        image_files = os.listdir(episode_images_dir)
        impage_path_last = None
        for image_file in image_files:
            image_path = os.path.join(episode_images_dir, image_file)
            measurement_path = os.path.join(episode_measurements_dir, os.path.splitext(image_file)[0])
            try:
                if pairs:
                    if check_is_following(image_path_last, image_path):
                        result.append([image_path, image_path_last, *pitch, *get_car_info(measurement_path)])
                else:
                    result.append([image_path, *pitch, *get_car_info(measurement_path)])
                image_path_last = image_path
            except Exception as e:
                print('error loading {}: {}'.format(image_file, str(e)))
                image_path_last = None
            
        res = np.array(result)
        return res

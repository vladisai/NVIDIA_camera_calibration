import os
import configparser
import json
import itertools
import numpy as np
from matplotlib.pyplot import imshow
import PIL
import random
import time

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

config_path = 'configs/000000_config'
measurements_dir = 'measurements'

def read_config(config_path):
    config = configparser.ConfigParser()
    with open(config_path) as fp:
        config.read_file(itertools.chain(['[global]'], fp), source=config_path)
    d = config['global']
    return np.float64(d['pitch'])
    #return np.array(list(map(float, [d['roll'], d['pitch'], d['yaw']])))

def read_image(image_path, shape=(160, 120, 1)):
    w, h, c = shape
    img = load_img(image_path)
    if c == 1:
        img = img.convert('L')
    if img.width != w or img.height != h:
        img = img.resize((w, h),PIL.Image.ANTIALIAS)
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

def get_car_info(measurements_path):
    json_file = open(measurements_path)
    json_str = json_file.read()
    json_data = json.loads(json_str)
    speed = json_data['playerMeasurements']['forwardSpeed']
    steer = json_data['playerMeasurements']['autopilotControl']['steer']
    throttle = json_data['playerMeasurements']['autopilotControl']['throttle']
    return np.float64(speed), np.float64(steer), np.float64(throttle)

read_image_vec = np.vectorize(read_image)

class Dataset:
    def __init__(self, path, camera_dir, limit = -1):
        dirs = os.listdir(path)
        random.shuffle(dirs)
        self.dataset = []
        for d in dirs:
            self.dataset.extend(self.add_episode(os.path.join(path, d), camera_dir))
            if limit > 0 and len(self.dataset) > limit:
                break
        self.dataset = np.stack(np.array(self.dataset))
    
    def get_dataset(self):
        return self.dataset

    def add_episode(self, episode_path, camera_dir):
        result = []
        episode_images_dir = os.path.join(episode_path, camera_dir)
        episode_measurements_dir = os.path.join(episode_path, measurements_dir)

        episode_config_path = os.path.join(episode_path, config_path)
        pitch = read_config(episode_config_path)

        image_files = os.listdir(episode_images_dir)
        for image_file in image_files:
            image_path = os.path.join(episode_images_dir, image_file)
            measurement_path = os.path.join(episode_measurements_dir, os.path.splitext(image_file)[0])
            try:
                result.append([image_path, pitch, *get_car_info(measurement_path)])
            except:
                print('error reading steer from json for {}'.format(image_path))
                pass
        return np.array(result)

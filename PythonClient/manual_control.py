#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Keyboard controlling for CARLA. Please refer to client_example.py for a simpler
# and more documented example.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    R            : restart level

STARTING in a moment...
"""

from __future__ import print_function

from google.protobuf.json_format import MessageToJson

import argparse
import logging
import random
import time
import numpy as np
import os
from python_files.models import ModelFromFile

from PIL import Image

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_i
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line


WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
MINI_WINDOW_WIDTH = 160
MINI_WINDOW_HEIGHT = 120


def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        PlayerVehicle='/Game/Blueprints/Vehicles/AudiTT/AudiTT.AudiTT_C',
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=False,
        NumberOfVehicles=0,
        NumberOfPedestrians=0,
        WeatherId=0,
        QualityLevel='Low')
    settings.randomize_seeds()
    camera_main = sensor.Camera('CameraRGB_main')
    camera_main.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera_main.set_position(-9.0, 0.0, 3.1)
    camera_main.set_rotation(-10.0, 0.0, 0.0)
    camera0 = sensor.Camera('CameraRGB_clean')
    camera0.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    camera0.set_position(2.0, 0.0, 1.1)
    camera0.set_rotation(0.0, 0.0, 0.0)
    # camera1 = sensor.Camera('CameraRGB')
    # camera1.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    # camera1.set_position(-6.0, 0.0, 2.5)
    # camera1.set_rotation(0.0, 0.0, 0.0)
    settings.add_sensor(camera0)
    settings.add_sensor(camera_main)
    # settings.add_sensor(camera1)
    # camera1 = sensor.Camera('CameraDepth', PostProcessing='Depth')
    # camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    # camera1.set_position(2.0, 0.0, 1.4)
    # camera1.set_rotation(0.0, 0.0, 0.0)
    # settings.add_sensor(camera1)
    # camera2 = sensor.Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
    # camera2.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    # camera2.set_position(2.0, 0.0, 1.4)
    # camera2.set_rotation(0.0, 0.0, 0.0)
    # settings.add_sensor(camera2)
    # if args.lidar:
    #     lidar = sensor.Lidar('Lidar32')
    #     lidar.set_position(0, 0, 2.5)
    #     lidar.set_rotation(0, 0, 0)
    #     lidar.set(
    #         Channels=32,
    #         Range=50,
    #         PointsPerSecond=100000,
    #         RotationFrequency=10,
    #         UpperFovLimit=10,
    #         LowerFovLimit=-30)
    #     settings.add_sensor(lidar)
    return settings


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)

        self._timer = None
        self._display = None
        self._main_image = None
        self._inf_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
        self._enable_autopilot = args.autopilot
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = False
        self._city_name = args.map_name
        self._map = CarlaMap(self._city_name, 0.1643, 50.0) if self._city_name is not None else None
        self._map_shape = self._map.map_image.shape if self._city_name is not None else None
        self._map_view = self._map.get_map(WINDOW_HEIGHT) if self._city_name is not None else None
        self._position = None
        self._agent_positions = None

        if args.model != None:
            self._model = ModelFromFile(args.model)
            self._enable_autopilot = True
        else:
            self._model = None

        self._save_images_to_disk = args.save_images_to_disk
        self._out_filename_format = args.out_filename_format
        self._episode = 0

    def execute(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                self._on_loop()
                self._on_render()
        finally:
            pygame.quit()

    def _initialize_game(self):
        if self._city_name is not None:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + int((WINDOW_HEIGHT/float(self._map.map_image.shape[0]))*self._map.map_image.shape[1]), WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        logging.debug('pygame started')
        self._on_new_episode()

    def _on_new_episode(self):
        settings = self._carla_settings
        camera1 = sensor.Camera('CameraRGB_pitch')
        camera1.set_image_size(160, 120)
        camera1.set_position(2.00, 0.0, 1.10)
        pitch = random.uniform(-15, 15)
        camera1.set_rotation(pitch=pitch, yaw=0, roll=0)
        settings.add_sensor(camera1)

        scene = self.client.load_settings(settings)
        number_of_player_starts = len(scene.player_start_spots)
        player_start = np.random.randint(number_of_player_starts)
        print('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False
        self._episode += 0
        self._frame = 0

        if self._save_images_to_disk:
            conf = self._out_filename_format.format(self._episode, 'configs', 0) + '_config'
            if not os.path.exists(os.path.dirname(conf)):
                try:
                    os.makedirs(os.path.dirname(conf))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            with open(conf, 'w') as f:
                f.write('pitch={}\n'.format(pitch))
                f.write('roll={}\n'.format(roll))
                f.write('yaw={}\n'.format(yaw))

    def _on_loop(self):
        self._timer.tick()
        self._frame += 1

        measurements, sensor_data = self.client.read_data()

        self._main_image = sensor_data.get('CameraRGB_main', None)
        self._inf_image = sensor_data.get('CameraRGB_clean', None)

        if self._save_images_to_disk:
            for name, measurement in sensor_data.items():
                if name != 'CameraRGB_main':
                    filename = self._out_filename_format.format(self._episode, name, self._frame)
                    measurement.save_to_disk(filename)

            filename = self._out_filename_format.format(self._episode, 'measurements', self._frame)
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            with open(filename, 'w') as f:
                f.write(MessageToJson(measurements))

        # Print measurements every second.
        if self._timer.elapsed_seconds_since_lap() > 1.0:
            if self._city_name is not None:
                # Function to get car position on map.
                map_position = self._map.convert_to_pixel([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                # Function to get orientation of the road car is in.
                lane_orientation = self._map.get_lane_orientation([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])

                self._print_player_measurements_map(
                    measurements.player_measurements,
                    map_position,
                    lane_orientation)
            else:
                self._print_player_measurements(measurements.player_measurements)

            # Plot position on the map as well.

            self._timer.lap()

        control = self._get_keyboard_control(pygame.key.get_pressed())
        # Set the player position
        if self._city_name is not None:
            self._position = self._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self._agent_positions = measurements.non_player_agents

        if control is None:
            self._on_new_episode()
        elif self._enable_autopilot:
            prediction = 0
            autopilot_control = measurements.player_measurements.autopilot_control
            if self._model is not None:
                array = image_converter.to_rgb_array(self._inf_image)
                npa = np.array(array)
                img = Image.fromarray(array).convert('L')
                npa = np.asarray(img)
                npa = npa.astype(np.float64) / 255
                npa = np.expand_dims(npa, axis = 3)
                meta = np.array([measurements.player_measurements.forward_speed])[:, np.newaxis]
                image = np.expand_dims(npa, axis = 0)
                inputs = [image]
                prediction = self._model.predict(inputs)
                autopilot_control.steer = prediction[0][0]
                #autopilot_control.throttle = prediction[0][1]
                print('predicted controls are {}'.format(prediction[0]))
            
            self.client.send_control(autopilot_control)
        else:
            self.client.send_control(control)

    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        if keys[K_r]:
            return None
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            control.steer = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            control.steer = 1.0
        if keys[K_UP] or keys[K_w]:
            control.throttle = 1.0
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        if keys[K_p]:
            self._enable_autopilot = not self._enable_autopilot
        if keys[K_i]:
            self._save_images_to_disk = not self._save_images_to_disk
        control.reverse = self._is_on_reverse
        return control

    def _print_player_measurements_map(
            self,
            player_measurements,
            map_position,
            lane_orientation):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += 'Map Position ({map_x:.1f},{map_y:.1f}) '
        message += 'Lane Orientation ({ori_x:.1f},{ori_y:.1f}) '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            map_x=map_position[0],
            map_y=map_position[1],
            ori_x=lane_orientation[0],
            ori_y=lane_orientation[1],
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _print_player_measurements(self, player_measurements):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

    def _on_render(self):
        gap_x = (WINDOW_WIDTH - 2 * MINI_WINDOW_WIDTH) / 3
        mini_image_y = WINDOW_HEIGHT - MINI_WINDOW_HEIGHT - gap_x

        if self._main_image is not None:
            array = image_converter.to_rgb_array(self._main_image)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))

        if self._inf_image is not None:
            array = image_converter.to_rgb_array(self._inf_image)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self._display.blit(surface, (gap_x, mini_image_y))

        # if self._mini_view_image2 is not None:
        #     array = image_converter.labels_to_cityscapes_palette(
        #         self._mini_view_image2)
        #     surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        #     self._display.blit(
        #         surface, (2 * gap_x + MINI_WINDOW_WIDTH, mini_image_y))

        # if self._lidar_measurement is not None:
        #     lidar_data = np.array(self._lidar_measurement.data[:, :2])
        #     lidar_data *= 2.0
        #     lidar_data += 100.0
        #     lidar_data = np.fabs(lidar_data)
        #     lidar_data = lidar_data.astype(np.int32)
        #     lidar_data = np.reshape(lidar_data, (-1, 2))
        #     #draw lidar
        #     lidar_img_size = (200, 200, 3)
        #     lidar_img = np.zeros(lidar_img_size)
        #     lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        #     surface = pygame.surfarray.make_surface(lidar_img)
        #     self._display.blit(surface, (10, 10))

        # if self._map_view is not None:
        #     array = self._map_view
        #     array = array[:, :, :3]

        #     new_window_width = \
        #         (float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
        #         float(self._map_shape[1])
        #     surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        #     w_pos = int(self._position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
        #     h_pos = int(self._position[1] *(new_window_width/float(self._map_shape[1])))

        #     pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)
        #     for agent in self._agent_positions:
        #         if agent.HasField('vehicle'):
        #             agent_position = self._map.convert_to_pixel([
        #                 agent.vehicle.transform.location.x,
        #                 agent.vehicle.transform.location.y,
        #                 agent.vehicle.transform.location.z])

        #             w_pos = int(agent_position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
        #             h_pos = int(agent_position[1] *(new_window_width/float(self._map_shape[1])))

        #             pygame.draw.circle(surface, [255, 0, 255, 255], (w_pos, h_pos), 4, 0)

        #     self._display.blit(surface, (WINDOW_WIDTH, 0))


        pygame.display.flip()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-m', '--map-name',
        metavar='M',
        default=None,
        help='plot the map of the current city (needs to match active map in '
             'server, options: Town01 or Town02)')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '--model',
        dest='model',
        default=None,
        help='model name to load')
    argparser.add_argument(
        '--test',
        action='store_true',
        dest='test',
        help='test model, output nothing but frames time to collision')

    args = argparser.parse_args()
    args.out_filename_format = 'out/episode_{:0>4d}/{:s}/{:0>6d}'

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args)
                game.execute()
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

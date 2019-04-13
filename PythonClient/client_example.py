#Low!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client example."""

from __future__ import print_function

from google.protobuf.json_format import MessageToJson

import argparse
import logging
import random
import time
import os
import json

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

last_saved = -1

def run_carla_client(args):
    # Here we will run 3 episodes with 300 frames each.
    number_of_episodes = args.episodes
    frames_per_episode = args.frames

    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')

        for episode in range(0, number_of_episodes):
            # Start a new episode.

            pitch = random.uniform(-15, 15)
            roll = random.uniform(-15, 15)
            yaw = random.uniform(-15, 15)

            if args.settings_filepath is None:

                # Create a CarlaSettings object. This object is a wrapper around
                # the CarlaSettings.ini file. Here we set the configuration we
                # want for the new episode.
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

                # Now we want to add a couple of cameras to the player vehicle.
                # We will collect the images produced by these cameras every
                # frame.

                # The default camera captures RGB images of the scene.
                camera0 = Camera('CameraRGB_clean')
                camera0.set_image_size(320, 180)
                camera0.set_position(2.00, 0.0, 1.10)

                camera1 = Camera('CameraRGB_clean_small')
                camera1.set_image_size(160, 120)
                camera1.set_position(2.00, 0.0, 1.10)

                camera1_1 = Camera('CameraRGB_clean_shiftp')
                camera1_1.set_image_size(160, 120)
                camera1_1.set_position(2.00, 0.3, 1.10)

                camera1_2 = Camera('CameraRGB_clean_shiftn')
                camera1_2.set_image_size(160, 120)
                camera1_2.set_position(2.00, -0.3, 1.10)
                 
                camera2 = Camera('CameraRGB_roll_pitch')
                camera2.set_image_size(160, 120)
                camera2.set_position(2.00, 0.0, 1.10)
                camera2.set_rotation(pitch=pitch, yaw=0, roll=roll)
                 
                camera3 = Camera('CameraRGB_roll_pitch_yaw')
                camera3.set_image_size(160, 120)
                camera3.set_position(2.00, 0.0, 1.10)
                camera3.set_rotation(pitch=pitch, yaw=yaw, roll=roll)

                settings.add_sensor(camera0)
                settings.add_sensor(camera1)
                # settings.add_sensor(camera1_1)
                # settings.add_sensor(camera1_2)
                # settings.add_sensor(camera2)
                # settings.add_sensor(camera3)

            else:

                # Alternatively, we can load these settings from a file.
                with open(args.settings_filepath, 'r') as fp:
                    settings = fp.read()

            # Now we load these settings into the server. The server replies
            # with a scene description containing the available start spots for
            # the player. Here we can provide a CarlaSettings object or a
            # CarlaSettings.ini file as string.
            scene = client.load_settings(settings)

            # Choose one player start at random.
            number_of_player_starts = len(scene.player_start_spots)
            player_start = random.randint(0, max(0, number_of_player_starts - 1))

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode...')
            client.start_episode(player_start)
              # Save the images to disk if requested.
             # Iterate every frame in the episode.
            for frame in range(0, frames_per_episode):

                # Read the data produced by the server this frame.
                measurements, sensor_data = client.read_data()

                # Print some of the measurements.
                print_measurements(measurements)

                def should_save(frame, measurements):
                    global last_saved
                    if frame < 40:
                        return False
                    if last_saved == frame - 5:
                        print('last saved')
                        last_saved = -100
                        return True
                    if measurements.player_measurements.forward_speed < 1:
                        if random.random() < 0.01:
                            print('random slow')
                            last_saved = frame
                            return True
                        else:
                            return False
                    if frame % 50 == 0:
                        print('50')
                        last_saved = frame
                        return True
                    if random.random() < abs(measurements.player_measurements.autopilot_control.steer) ** 2 * 10:
                        print('steer {}'.format(measurements.player_measurements.autopilot_control.steer))
                        last_saved = frame
                        return True
                    else:
                        return False
                    return False

                # Save the images to disk if requested.
                if args.save_images_to_disk:
                    if args.save_all or should_save(frame, measurements):
                        print('saving frame ', frame)
                        for name, measurement in sensor_data.items():
                            filename = args.out_filename_format.format(episode, name, frame)
                            measurement.save_to_disk(filename)
                        filename = args.out_filename_format.format(episode, 'measurements', frame)
                        if not os.path.exists(os.path.dirname(filename)):
                            try:
                                os.makedirs(os.path.dirname(filename))
                            except OSError as exc: # Guard against race condition
                                if exc.errno != errno.EEXIST:
                                    raise
                        with open(filename, 'w') as f:
                            f.write(MessageToJson(measurements))


                # We can access the encoded data of a given image as numpy
                # array using its "data" property. For instance, to get the
                # depth value (normalized) at pixel X, Y
                #
                #     depth_array = sensor_data['CameraDepth'].data
                #     value_at_pixel = depth_array[Y, X]
                #

                # Now we have to send the instructions to control the vehicle.
                # If we are in synchronous mode the server will pause the
                # simulation until we send this control.

                if not args.autopilot:

                    client.send_control(
                        steer=random.uniform(-1.0, 1.0),
                        throttle=0.5,
                        brake=0.0,
                        hand_brake=False,
                        reverse=False)

                else:

                    # Together with the measurements, the server has sent the
                    # control that the in-game autopilot would do this frame. We
                    # can enable autopilot by sending back this control to the
                    # server. We can modify it if wanted, here for instance we
                    # will add some noise to the steer.

                    control = measurements.player_measurements.autopilot_control
                    control.steer += random.uniform(-0.1, 0.1)
                    client.send_control(control)

            conf = args.out_filename_format.format(episode, 'configs', 0) + '_config'

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


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.0f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
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
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    argparser.add_argument(
        '--save_all',
        default=False,
        action='store_true',
        help='save all images')
    argparser.add_argument(
        '--episodes',
        default=600,
        type=int,
        help='Number of episodes to record')
    argparser.add_argument(
        '--frames',
        default=1800,
        type=int,
        help='Number of frames per epiosode to record')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = 'out/episode_{:0>4d}/{:s}/{:0>6d}'

    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

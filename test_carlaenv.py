from __future__ import print_function
import os
import numpy as np
import argparse
import logging
import random
import time
import collections
import random
from carla import CarlaEnv

from carla_utils.client import make_carla_client
from carla_utils.sensor import Camera, Lidar
from carla_utils.settings import CarlaSettings
from carla_utils.tcp import TCPConnectionError
from carla_utils.util import print_over_same_line


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
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')

    argparser.add_argument('--gpu_id', default=1)
    args = argparser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'
    
    vehicles = ['AudiA2', 'AudiTT', 'BmwGrandTourer', 'BmwIsetta', 'CarlaCola', 
            'ChevroletImpala', 'CitroenC3', 'DodgeChargePolice', 'JeepWranglerRubicon', 
            'Mercedes','Mini', 'Mustang', 'NissanMicra', 'NissanPatrol']
    while True:
        try:
            
            with make_carla_client(args.host, args.port) as client :
                obs_config = {
                        'p_transform':False, # dim 6
                        'p_acceleration':False, # dim 3
                        'p_forward_speed':False,
                        'p_collision_vehicles':False,
                        'p_collision_pedestrians':False,
                        'p_collision_other':False,
                        'p_intersection_otherlane':False,
                        'p_intersection_offroad':False,
                        'camera':True,
                        'vehicles':False,
                        'pedestrians':False,
                        'traffic_lights':False,
                        'speed_limit_signs':False,
                        'cropped':True}
                env = CarlaEnv(client, obs_config,51,starting_point=33)
                # add a sensor 
                env.set_map(nv=0, np=0, weather=1, sync=True, send_np_info = False, quality=True )
                env.set_vehicle('Mustang')
                # add a sensor 
                env.set_sensor("RGB",size=[256,256], camera_type=None)
                env.make(init_step=90, init_action=[0,1,0,0,0])
                for k in range(5):
                    env.reset()
                    for j in range(50) :
                        if j < 11  :
                            action = [0,1,0,0,0]
                        elif j < 25+2*k :
                            action = [-1,1,1,1,0]
                        else :
                            action = [0,1,0,0,0]
                        obs,_,done,_ = env.step(action)
                    env.make_video(str(k))
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


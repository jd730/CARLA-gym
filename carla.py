from __future__ import print_function

import os
import numpy as np
from gym.spaces import Box
from gym import utils
import cv2
import skvideo.io  
import argparse
import logging
import random
import time
import collections
import math
from carla_utils.client import make_carla_client
from carla_utils.sensor import Camera, Lidar
from carla_utils.settings import CarlaSettings
from carla_utils.tcp import TCPConnectionError
from carla_utils.util import print_over_same_line
from carla_obs import CarlaObs

#from ..common.utils import get_file_from_asset_dir 
# from ..agents.Rarm import Rarm
import scipy.misc
import moviepy.editor as mpy
from scipy.misc import imresize

    """
        Carla Action
            steer : float [-1.0,1.0], degree, depnding on type of vehicle, the maximum steer can be changed. e.g. Mustang is 70 degrees
            throttle : float [0,1.0], throttle input
            brake : float [0,1.0], brake input
            hand_brake : bool, hand brake
            reverse : bool, reverse gear
    """

class CarlaEnv(object) :
    """
        Environment of CARLA

        Args :
            client : client of carla agent
            max_lengths : maximum steps of each episode
            starting_point : starting position number of map
            setting_path : path of setting file which format is usually .ini

        Parameters :
            timer : Timer class
            num_episode : float, current episode number.
            nframe_passed : float, steps of current episode.
            obs :CarlaObs, observations.
            action_space : Box, dimension of action space
            observation_space : Box, dimension of observation space
            max_length : maximum steps of each episode
            obs_space_change_flag : boolean, if True, observation_space is changed at reset()
            
            filepath : for make_video. It will be removed because we do not need make_video.
            camera : dictionary of image sequence. It will be removed due to same reason

        Func :
            set_map : set map environment
            set_vehicle : set type of player vehicle
            set_sensor : set sensor, there are 4 kinds of sensor.
            set_obs_config : set observation config file
            seed : set seed
            step : take one step using given action
            _step : give an action and take observations from server
            reward : return reward according to observation.
            done : check whether current episode is done or not
            get_obs : get observation,
            render : get dictionary of camera
            reset : make new episode
            make : make new environment

            _add_camera : convert obs to cameras. It will be removed
    """
    def __init__(self, client,obs_config, max_length, starting_point=110, setting_path=None, env_config={}, filepath='test') :
        self.client = client
        self.obs_config = obs_config
        self.starting_point =starting_point
        if setting_path is None :
            self.settings = CarlaSettings()
        else : 
            with open(setting_path, 'r') as fp:
                self.settings = fp.read()
        self._timer = None
        self.reset_timer = 0
        self.name = env_config['name'] if 'name' in env_config else 'carla'
        self.num_episode = 0
        self.step_counter = 0
        self.step_timer = 0
        self.t = 0 # never init
        self.nframe_passed = 0
        self.filepath = filepath
        if os.path.exists(self.filepath)==False :
            os.mkdir(self.filepath)

        self.cameras = {}
        self.obs = None
        self.pseudo_ob = None
        self.action_space=None
        self.observation_space=None
        self.max_length = max_length
        # when obs_config or sensor is changed, dimension of observation space is changed
        self.obs_space_change_flag = True
        self.load_setting_flag = False
        self.max_speed = 0

        self.init_step = 0
        self.init_action = None

        # for evaluation
        self.turn_ed = -1
        self.turn_st = -1
        self.achieve_tpoint = False

        self.logs = []
        self.alogs = []

        # For fitting gym setting and using Monitor Wrapper
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {'render.modes' : []}
        self._set_action_space()
        self.spec = None
        self.num_envs = 1
        self.horizon = env_config['horizon'] if 'horizon' in env_config else 2048
        self.dense_reward = env_config['dense_reward'] if'dense_reward' in env_config else False
        self.goal_reward = env_config['goal_reward'] if 'goal_reward' in env_config else 20
        
        self.next_force = True
 



    def set_map(self,nv=0,np=0, weather=None, sync=True, send_np_info=False,quality=False) :
        """
            Set map environment

            Args :
                nv : int, number of non player vehicles
                np : int, number of non player players
                weather : int or None : weather conditions. If None, randomly generated
                    1 - ClearNoon                2 - CloudyNoon               3 - WetNoon
                    4 - WetCloudyNoon            5 - MidRainyNoon             6 - HardRainNoon
                    7 - SoftRainNoon             8 - ClearSunset              9 - CloudySunset
                    10 - WetSunset              11 - WetCloudySunset         12 - MidRainSunset
                    13 - HardRainSunset         14 - SoftRainSunset
                sync :boolean, In synchronous mode, CARLA waits every frame until the control from the client is received.
                        It should be  True
                send_np_info : Send info about every non-player agent in the scene every frame, 
                    the information is attached to the measurements message. 
                    This includes other vehicles, pedestrians and traffic signs. Disabled by default to improve performance.
                quality : boolean True : High, False : Low
                    Quality level of the graphics, a lower level makes the simulation run considerable faster
        """
        if quality :
            q = True
        else :
            q = False
        if weather is None :
            self.settings.randomize_weather()
        else :
            self.settings.set(WeatherId=weather)
            self.settings.set(
                SynchronousMode=sync,
                SendNonPlayerAgentsInfo=send_np_info,
                NumberOfVehicles=nv,
                NumberOfPedestrians=np,
                QualityLevel=q)
        #random seeds for vehicles and pedestrians
        self.settings.randomize_seeds()

    def set_vehicle(self, name="Mustang") :
        """
            Set type of vehicle
            Args :
                name : String or Integer
        """

        car_type = ["AuidiA2", "AudiTT", "BMX Grand Tourer", "BMW Isetta", "CarlaCola",
                "CitroenC3", "Chevrolet Impala", "DodgeCharger Police", "Jeep WranglerRubicon", 
                "Mercede","Mini", "Mustang", "Nissan Micra", "Nissan Patrol", "SeatLeon", 
                "ToyotaPrius", "VolkswagenT2"]
        if type(name) is int :
            name = car_type(name)
        self.vehicle_name = name
        path = "/Game/Blueprints/Vehicles/{}/{}.{}_C".format(name,name,name)
        self.settings.set(PlayerVehicle=path)


    def set_sensor(self, name="View", pos=[-10,0,3], rot=[0,0,-0.3], 
            size=[256,256], camera_type=None,fov=90, lidar_info=None) :
        """
            Set mew sensor
            
            Args :
                name
                size : Size of the captured image in pixels.
                pos : list of float dim(3), Position of the camera relative to the car in meters.
                rot : list of float dim(3), Rotation of the camera relative to the car in degrees.
                camera_type :
                    * None                  No effects applied.
                    * SceneFinal            Post-processing present at scene (bloom, fog, etc).
                    * Depth                 Depth map ground-truth only.
                    * SemanticSegmentation  Semantic segmentation ground-truth only.
                    * Lidar                 Lidar
          
                fov :  Camera (horizontal) field of view in degrees.

                lidar_info : list of information of Lidar
 
        """
        if camera_type in [None, "SceneFinal", "Depth", "SemanticSegmentation"] :
            camera = Camera(name, PostProcessing=camera_type)
            camera.set_image_size(size[0], size[1])
            camera.set(FOV=fov)
        elif camera_type is "Lidar" :
            camera = Lidar(name)
            if lidar_info is None :
                lidar_info = [32,50,100000,10,10,-30]
            camera.set(
                Channels=lidar_info[0],
                Range=lidar_info[1],
                PointsPerSecond=lidar_info[2],
                RotationFrequency=lidar_info[3],
                UpperFovLimit=lidar_info[4],
                LowerFovLimit=lidar_info[5])
        else :
            raise ValueError ("{} type does not exist in Sensor".format(camera_type))
        # Set its position and rotation relative to the car in meters.
        camera.set_position(pos[0], pos[1], pos[2])
        camera.set_rotation(rot[0], rot[1], rot[2])
        # Set image resolution in pixels.
        self.settings.add_sensor(camera)
        self.obs_space_change_flag = True

    def set_obs_config(self, obs_config) :
        self.obs_config = obs_config
        self.obs_space_chang_flat = True

    def _set_action_space (self):
        high = np.array([1,1,1,1,1]) 
        low = np.array([-1,0,0,0,0])
        self.action_space = Box(low,high)

    def seed(self,s=0) :
        return _seed(s)

    def _seed(self, seed=None):
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]
    @property
    def get_obs(self):
        return self.obs.conv_obs

    def make(self, init_step=40, init_action=[0,0,0,0,0], default=False):
        """
            Make a new episode

            Args : 
                init_step : the number of steps doing init_action
                init_action : initial action for starting agent with some speed
                default : bool [True], if it is true, we used default setting
        """
        self.nframe_passed = 0
        self.init_step=init_step
        self.init_action=init_action
        self.settings.randomize_seeds()
       
        if default :
            self.set_map(0,0,1)
            self.set_vehicle()
#            self.set_sensor("RGB", camera_type="SceneFinal")
#            self.set_sensor("RGB")
#            self.set_sensor("RGB_non_proc")
#            self.set_sensor("Depth",camera_type="Depth")
#            self.set_sensor("Seg",camera_type="SemanticSegmentation")
#            self.set_sensor("Lidar",camera_type="Lidar")

        # this is for getting observation space
        self.reset()

        
    
    def reset(self) :
        """
            Reset the episode

            Return :
                obs, rewards, done
        """
        
        self.num_episode += 1
        start = time.time()

        if self.load_setting_flag is False :
            scene = self.client.load_settings(self.settings) # we will not chage at all
            self.load_setting_flag = True
            number_of_player_starts = len(scene.player_start_spots)
            player_start = np.random.randint(number_of_player_starts)

        self.nframe_passed = 0       
        self.obs = None
        
        self.turn_st = -1
        self.turn_ed = -1
        self.achieve_tpoint = False
        self.logs = []
        self.alogs = []
        self.cameras = {}
        self.client.start_episode(self.starting_point) # The list of player starts is retrived by "load_settings"
        
        if self.obs_space_change_flag :
            measurements, sensor_data = self._step([0,0,0,0,0])
            next_obs = CarlaObs(measurements, sensor_data, self.obs_config)
            self.observation_space = next_obs.observation_space
            self.obs = next_obs
            self.calc_pseudo_ob()
            self.obs_space_change_flag = False
            return next_obs.conv_obs

        # Since the agent is starting on the air, first several frames are wait for that period.
        for _ in range(47):
            self._step([0,0,0,0,0],True)

        # initial movement
        for _ in range(self.init_step-5):
            measurements, sensor_data = self._step(self.init_action,True)
        
        for _ in range(5):
            measurements, sensor_data = self._step([0,1-np.random.uniform(1e-3),0,0,0],pert=True)

        next_obs = CarlaObs(measurements, sensor_data, self.obs_config)
        self.obs = next_obs
        conv_obs = self.obs.conv_obs
        self.reset_timer += time.time() - start
        print('{} Starting new episode...{} {}'.format(self.num_episode,self.client.port, time.time()-start))
        return conv_obs

   
    def render(self, mode='rgb_array', close=False) :
        """
            Render an image. In case of CARLA, we already have rendered image, sensor_data
            Args :
                mode : 'rgb_array',
                close : dummy for gym
            Returns : images (np array)
        """
        for k in self.obs.camera.keys() :
            img = self.obs.camera[k]
            return img

    def _step(self, action, pert=False) :
        """
            Send an action and get data from CARLA server

            Args :
                action, np array [5]
            Returns :
                measurements, and sensor_data (dictionary)
        """
        self.client.send_control(
                steer=action[0],
                throttle= action[1],
                brake=action[2],
                hand_brake=action[3], # share with brake
                reverse=action[4]
            )
        measurements, sensor_data = self.client.read_data()
        self.t += 1
        return measurements, sensor_data


    def step(self,action) :
        """
            Take a step using given action.

            Args :
                action : CarlaAction
            Return :
                next_obs : np array
                rewards : float
                done : boolean
                infos : empty dictionary for fitting gym setting
        """
        # Act
        self.step_counter += 1
        self.step_timer -= time.time()
        self.alogs.append(action)
        
        # get observation
        measurements, sensor_data = self._step(action)
        next_obs = CarlaObs(measurements, sensor_data, self.obs_config)
        self._add_cameras(next_obs)
        self.obs = next_obs
        reward, done,infos = self.goal_rewards()
        if infos is None :
            infos = {}
        conv_obs = next_obs.conv_obs
        self.step_timer += time.time()
        
        self.eval()
        self.write_log()
        self.calc_pseudo_ob()

        return conv_obs, reward, done, infos # dummy output


    @property
    def done(self) :
        """
            Judge current reward by using given observations

            Return : boolean
        """
        if self.max_length == self.nframe_passed :
            return True
        return False

    def check_goal(self) :
        """
            Check whether the agent achieves the goal state or not.

            Return : boolean
        """
        loc = self.obs.p_transform.location
        rot = self.obs.p_transform.rotation
        pos = (loc.x > 11) and loc.y > 324 and loc.y < 331 # 11m judge wheter it turn right or not, 3m is boundary of road
        direction = abs(rot.yaw) < 30 # 30 degrees 0 is perpendicular.
        speed = (self.obs.p_forward_speed > 5)
        end = (self.max_length == self.nframe_passed)

        return (pos and direction and speed and end)
        




    def goal_rewards(self, infos={}, agent_done=None) :
        """
            Judge current reward by using current observations

            Return : reward, float
        """

        # increment step counter
        self.nframe_passed += 1
        done = False
        reward = 0

        if self.dense_reward :
            reward = self.obs.p_forward_speed # 0 - 100 usually 0~40
            reward -= 0.001 * abs(self.obs.p_collision_max) # 0 - 10k
            reward -= 10 * (self.obs.p_intersection_otherlane + self.obs.p_intersection_offroad) # 0 - 1
            if  self.obs.p_forward_speed < 10 :
                reward -= (10 - self.obs.p_forward_speed) * 10

        if self.check_goal() : # success
            done = True
            reward = self.goal_reward
            if infos is None :
                infos = {}
            elif infos is not {} :
                for info in infos :
                   info['score'] = True
        # failure
        elif self.done :
            done = True
            reward = -self.goal_reward
        return reward, done, infos

    def calc_pseudo_ob (self) :
        loc = self.obs.p_transform.location
        rot = self.obs.p_transform.rotation
        acc = self.obs.p_acceleration
        spd = self.obs.p_forward_speed
        cols = [self.obs.p_collision_vehicles, self.obs.p_collision_pedestrians, self.obs.p_collision_other]
        ints = [self.obs.p_intersection_otherlane, self.obs.p_intersection_offroad]

        locs = [loc.x,loc.y,loc.z]
        rots = [rot.pitch,rot.yaw,rot.roll]
        accs = [acc.x,acc.y,acc.z]
        self.pseudo_ob =  locs + rots + accs + [spd] + cols + ints


    def write_log(self) :
       self.logs.append(self.pseudo_ob)

    # this is for the testing
    def make_video(self,name='') :
        filename = "{}/{}{}_{}".format(self.filepath,self.name,self.num_episode,name)
        for k in self.cameras.keys() :
            print(np.array(self.cameras[k]).shape)
            skvideo.io.vwrite(filename+'.mp4',np.array(self.cameras[k]))
            print ("MAKEING VIDEO" + filename)
        
        log = np.asarray(self.logs)
        alog = np.asarray(self.alogs)
        print(log.shape)
        print(alog.shape)

        np.savetxt(filename+'.txt',log,delimiter=',')
        np.savetxt(filename+'_a.txt',alog,delimiter=',')
        self.cameras={}

    def _add_cameras(self, ob) :
        for k in  ob.camera.keys():     
            if k not in self.cameras :
                self.cameras[k] = []
            self.cameras[k].append(ob.camera[k])

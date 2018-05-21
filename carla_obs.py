from __future__ import print_function

import os
import numpy as np
from gym.spaces import Box
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



# Vehicle and Pedestrian have same attributes, but both are separated in this code for better understanding
Vehicle = collections.namedtuple("Vehicle", ["id","forward_speed","transform","bounding_box_transform","bounding_box_extent"])
Pedestrian = collections.namedtuple("Pedestrian", ["id","forward_speed","transform","bounding_box_transform","bounding_box_extent"])
TrafficLight = collections.namedtuple("TrafficLight", ["id","transform","state"])
SpeedLimitSign = collections.namedtuple("SpeedLimitSign",["id","transform","speed_limit"])



class CarlaObs(object) :
    """
        Carla Observation which is combined with measurements and sensor_data
        
        Parementers :
            vehicles : list of non-player vehicles information.
            pedestrians : list of non-player pedestrians information.
            traffic_lights : list of non-player traffic_lights information.
            limit_signs : list of non-player limit_signs information.
            cameras : dictionary of camera sensors.
                key : sensor name
                value : ndarray
            p_transform : Transform, player's trnasform information.
            p_acceleration : Vector3D, m/s^2, player's acceleration.
            p_forward_speed : float, m/s
            p_collision_* : float, kg*m/s, collision intensity with vehicles, 
                            pedestrians, other
            p_intersection_* : float [0,1], percentage of intersection.

        Funcs :
            build_sensor_obs() : make camera dictionary.
            build_measurement_obs() : convert measurements to measurement related data.


    """
    def __init__ (self, measurements, sensor_data, config) :
        self.config = config

        self.measurements = measurements
        self.sensor_data = sensor_data
       
        # p_ : information of player
        self.p_transform = None
        self.p_acceleration = None
        self.p_forward_speed = 0

        # collision intensity kg*m/s
        self.p_collision_vehicles = 0
        self.p_collision_pedestrians = 0
        self.p_collision_other = 0
        self.p_collision_max = 0
        
        # percentage of intersection
        self.p_intersection_otherlane = 0
        self.p_intersection_offroad = 0

        self.camera = {}
        self.vehicles = []
        self.pedestrians = []
        self.traffic_lights = []
        self.speed_limit_signs = []
 

        self.cropped = config['cropped'] if 'cropped' in config else False
        self.build_sensor_obs()
        self.build_measurement_obs()

    def _resolve_transform(self, trans) :
        loc = trans.location
#        ori = trans.orientation # deprecated
        rot = trans.rotation
        rt = [loc.x,loc.y,loc.z]
#        rt += [ori.x,ori.y,ori.z]
        rt += [rot.pitch,rot.yaw,rot.roll]
        return rt
    
    def _resolve_Vector3D(self,vec) :
        return [vec.x, vec.y, vec.z]
    
    
    def recover_obs(self, conv_ob) :
        raise NotImplementedError
    
    @property
    def observation_space (self) :
        conv_ob=[]
        high = []
        low = []

        # np.append is much slower than list append https://stackoverflow.com/questions/29839350/numpy-append-vs-python-append
        flag = False
        h_vector = [math.inf, math.inf, math.inf]
        l_vector = [-math.inf, -math.inf, -math.inf]
        h_transform = [math.inf, math.inf, math.inf, 180, 180, 180]
        l_transform = [-math.inf, -math.inf, -math.inf, -180, -180, -180]
        h_vnp = [math.inf] +  h_transform * 2 + h_vector
        l_vnp = [-math.inf] + l_transform * 2 + l_vector
        if self.config['p_transform'] :
            high += h_transform
            low += l_transform
            flag = True
        if self.config['p_acceleration'] :
            high += h_vector
            low += l_vector
            flag = True
        if self.config['p_forward_speed'] :
            high.append(math.inf)
            low.append(-math.inf)
            flag = True
        if self.config['p_collision_vehicles'] :
            high.append(math.inf)
            low.append(-math.inf)
            flag = True
        if self.config['p_collision_pedestrians'] :
            high.append(math.inf)
            low.append(-math.inf)
            flag = True
        if self.config['p_collision_other'] :
            high.append(math.inf)
            low.append(-math.inf)
            flag = True
        if self.config['p_intersection_otherlane'] :
            high.append(1)
            low.append(0)
            flag = True
        if self.config['p_intersection_offroad'] :
            high.append(1)
            low.append(0)
            flag = True
        if self.config['vehicles'] :
            high += h_vnp * len(self.vehicles)
            low += l_vnp * len(self.vehicles)
            flag = True
        if self.config['pedestrians'] :
            high += h_vnp * len(self.pedestrians)
            low += l_vnp * len(self.pedestrians)
            flag = True
        if self.config['traffic_lights'] :
            h_traffics =(h_transform + [2]) * len(self.traffic_lights)
            l_traffics =(l_transform + [0]) * len(self.traffic_lights)
            flag = True
            high += h_traffics
            low  += l_traffics
        if self.config['speed_limit_signs'] :
            high += (h_transform + [math.inf]) * len(self.speed_limit_signs)
            low += (l_transform + [0]) * len(self.speed_limit_signs)
            flag = True
        if self.config['camera'] :
            for k in self.camera.keys() :
                img = self.camera[k]
                # due to depth and Lidar
                # depth [H,W] 0 ~ 1
                # Lidar [ ,?? ] ?? 
                if flag :
                    high += [math.inf] * (int(s[0]),int(s[1]),int(s[2]))# img.size
                    low += [-math.inf] * (int(s[0]),int(s[1]),int(s[2]))
                else :
                    s  = img.shape
                    high = np.inf * np.ones([int(s[0]),int(s[1]),int(s[2])])
                    low = -high
                high = np.array(high)
                low = np.array(low)
                return Box(high,low)

        return Box(np.array(high),np.array(low))

    @property
    def conv_obs (self) :
        """
            vehicles : list of non-player vehicles information.
            pedestrians : list of non-player pedestrians information.
            traffic_lights : list of non-player traffic_lights information.
            limit_signs : list of non-player limit_signs information.
            cameras : dictionary of camera sensors.
            key : sensor name
            value : ndarray
            p_transform : Transform, player's trnasform information.
            p_acceleration : Vector3D, m/s^2, player's acceleration.
            p_forward_speed : float, m/s
            p_collision_* : float, kg*m/s, collision intensity with vehicles, 
            pedestrians, other
            p_collision_max : float, kg*m/s, maximum absolute value of collisions
            p_intersection_* : float [0,1], percentage of intersection.
        """
        conv_ob = []
        flag = False
        if self.config['p_transform'] :
            conv_ob += self._resolve_transform(self.p_transform)
            flag = True
        if self.config['p_acceleration'] :
            conv_ob += self._resolve_Vector3D(self.p_acceleration)
            flag = True
        if self.config['p_forward_speed'] :
            conv_ob.append(self.p_forward_speed)
            flag = True
        if self.config['p_collision_vehicles'] :
            conv_ob.append(self.p_collision_vehicles)
            flag = True
        if self.config['p_collision_pedestrians'] :
            conv_ob.append(self.p_collision_pedestrians)
            flag = True
        if self.config['p_collision_other'] :
            conv_ob.append(self.p_collision_other)
            flag = True
        if self.config['p_intersection_otherlane'] :
            conv_ob.append(self.p_intersection_otherlane)
            flag = True
        if self.config['p_intersection_offroad'] :
            conv_ob.append(self.p_intersection_offroad)
            flag = True
        # id is not need for us
        if self.config['vehicles'] :
            flag = True
            for e in self.vehicles :
                conv_ob.append(e.forward_speed)
                conv_ob += self._resolve_transform(e.transform)
                conv_ob += self._resolve_transform(e.bounding_box_transform)
                conv_ob += self._resolve_Vector3D(e.bounding_box_extent)
        if self.config['pedestrians'] : 
            flag = True
            for e in self.pedestrians :
                conv_ob.append(e.forward_speed)
                conv_ob += self._resolve_transform(e.transform)
                conv_ob += self._resolve_transform(e.bounding_box_transform)
                conv_ob += self._resolve_Vector3D(e.bounding_box_extent)
        if self.config['traffic_lights'] :
            flag = True
            for tl in self.traffic_lights :
                conv_ob += self._resolve_transform(tl.transform)
                conv_ob.append(tl.state)
        if self.config['speed_limit_signs'] :
            flag = True
            for sls in self.speed_limit_signs :
                conv_ob += self._resolve_transform(sls.transform)
                conv_ob.append(sls.speed_limit)
        if self.config['camera'] :
            for k in self.camera.keys() :
                img = self.camera[k]
                s = img.shape
                if flag :
                    conv_ob = np.append(np.array(conv_ob) , cob.flatten()) # row-major
                else :
                    conv_ob = cob

        else :
            conv_ob = np.array(conv_ob)
        return conv_ob


    def build_measurement_obs(self) :
        # save player information
        p_measurement = self.measurements.player_measurements
        self.p_transform = p_measurement.transform
        self.p_acceleration = p_measurement.acceleration
        self.p_forward_speed = p_measurement.forward_speed
        self.p_collision_vehicles = p_measurement.collision_vehicles
        self.p_collision_pedestrians = p_measurement.collision_pedestrians
        self.p_collision_other = p_measurement.collision_other
        self.p_intersection_otherlane = p_measurement.intersection_otherlane
        self.p_intersection_offroad = p_measurement.intersection_offroad

        self.p_collision_max = max(abs(self.p_collision_vehicles), max(abs(self.p_collision_pedestrians),abs(self.p_collision_other)))
        # save non_player information
        for agent in self.measurements.non_player_agents :
            if agent.HasField('vehicle') :
                vehicle = Vehicle(id=agent.id,
                        forward_speed=agent.vehicle.forward_speed,
                        transform=agent.vehicle.transform,
                        bounding_box_transform=agent.vehicle.bounding_box.transform,
                        bounding_box_extent=agent.vehicle.bounding_box.extent)
                self.vehicles.append(vehicle)

            elif agent.HasField('pedestrian') :
                pedestrian = Pedestrian(id=agent.id,
                        forward_speed=agent.pedestrian.forward_speed,
                        transform=agent.pedestrian.transform,
                        bounding_box_transform=agent.pedestrian.bounding_box.transform,
                        bounding_box_extent=agent.pedestrian.bounding_box.extent)
                self.pedestrians.append(pedestrian)

            elif agent.HasField('traffic_light') :
                traffic_light = TrafficLight(id=agent.id,
                        transform=agent.traffic_light.transform,
                        state=agent.traffic_light.state)
                self.traffic_lights.append(traffic_light)

            elif agent.HasField('speed_limit_sign') :
                speed_limit_sign = SpeedLimitSign(id=agent.id,
                        transform=agent.speed_limit_sign.transform,
                        speed_limit=agent.speed_limit_sign.speed_limit)
                self.speed_limit_signs.append(speed_limit_sign)

            else :
                # Since there are four kinds of non_player agent, in this statement is error regin
                raise Exception ("there is a new non-player_agent or something wrong.")

    def build_sensor_obs(self) :
        # max 255 in case of RGB
        for name,measurement in self.sensor_data.items() :
            img = measurement.data
            self.camera[name] = img



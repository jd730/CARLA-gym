# CARLA
## How to run
### Boot a server
This commend includes a setting from `CarlaSettings.ini`
`sh /home/jd730/start.sh`

### Run a test code
`python3 /home/jdhwang/Dual_RL_CV/init_exps/share/ver3/exp_env/envs/carla_env/test_carlaenv.py`

### code in carla_env
#### carla.py
environment

#### test_carla_env.py
test code of carla environment

#### carla_utils/
copied from CARLA PythonClient libraries

## Attributes
### Actions
1. **steer** float, [-1.0, 1.0]. 
Maximum range depends of car (e.g. Mustang 70 degrees)
2. **throttle**
	float, [ 0.0, 1.0]
3. **brake**, float, [ 0.0, 1.0]
4.  **hand_brake** (BOOLEAN)
5. **reverse**  (BOOLEAN)

### Measurement
http://carla.readthedocs.io/en/latest/measurements/
#### Example
```
measurements, sensor_data = client.read_data()
for agent in measurements.non_player_agents:
    agent.id # unique id of the agent
    if agent.HasField('vehicle'):
        agent.vehicle.forward_speed
        agent.vehicle.transform
        agent.vehicle.bounding_box
```

#### Time-stemp
1. frame_number
2. platform_timestamp (ms)
3. game_timestemp (ms)

#### Playermeasurement
1. **transform**
	Transform, World transform of the player
2. **acceleration**
	Vector3D, m/s^2
3. **forward_speed**
	float, m/s
4. **collision_vehhicles** / **collision_pedestrians** / **collision_other**
	float kg*m/s
5. **intersection_otherlExane** / **intersection_offroad** 
	percentage flaot
6. **autopilot_control** (ignore)

#### Transform
x : direction of car
z : perpendicular with the ground
y : others

1. **location**
	Vector3D, m
2. **rotation**
	Rotation3D, degree
3. **orientation** (deprecated)

#### Non-player agents info
##### Vehicle/Pedestrian
1. id (UINT32)
2. **vehicle.forward_speed** (FLOAT) m/s
3. **vehicle.transform**  (TRANSFORM)
4. **vehicle.bounding_box.transform**  (TRANSFORM)
5. **vehicle.bounding_box.extent**  (VECTOR3D)
	Radii dimension of the bounding box in meters (in case of pedestrian, we assume all of them are same size)
	0,0,0 at right upper front

##### Traffic light
1. id
2. traffic_light.transform
3. traffic_light.state
	enum : GREEN, YELLOW, RED

##### Speed limit sign
1. id
2. speed_limit_sign.transform
3. speed_limit_sign.speed_limit  (float, m/s)


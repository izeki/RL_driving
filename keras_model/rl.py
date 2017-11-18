#parsing command line arguments
import math
import argparse
#matrix math
import numpy as np
#real-time server
import socketio
#concurrent networking 
import eventlet
#web server gateway interface
import eventlet.wsgi
from scipy.spatial.distance import euclidean
#threading
import threading
#Queue
from multiprocessing import Queue
#image manipulation
from PIL import Image
#web framework
from flask import Flask
#input output
from io import BytesIO
#decoding camera images
import base64
# RL class
from rl_agent import RLAgent
# utility function
import utils
# To solve "Not An Element Of Tensor Graph" bug
# https://justttry.github.io/justttry.github.io/not-an-element-of-Tensor-graph/
import tensorflow as tf
graph = None
from nets.DrivingPolicyNet import DrivingPolicyNet

# commandline arguments
args = None

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)

#set min/max speed for our autonomous car
MAX_SPEED = 60
MIN_SPEED = 10

#and a speed limit
speed_limit = MAX_SPEED

# immediate reward: only consider off road condition so far.
# off road: end state and reward = -1, reward = 0, otherwise.
reward = 0
is_first = True
is_off_road = False
is_replay = False

# RL agent
rlAgent = None


alpha1, alpha2 = 0.5, 0.5 #Used for calculating immediate reward

drive_policy_net = DrivingPolicyNet(utils.INPUT_SHAPE) ##Initialize drive_policy net ###
drive_policy_net.model_init()


#for command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

#Arguments are all current data received from unity
def rl_drive(image, curr_steer, curr_motor, speed, pitch, yaw, dist_left, dist_right):
    global graph, rlAgent, reward, args, is_first, is_off_road, is_replay

    #Format arguments for input to Driving Policy Net.
    IMU_arg = utils.format_metadata_RL(curr_steer, curr_motor, speed, pitch, yaw)
    dict_input = {'IMG_input': image, 'IMU_input': IMU_arg}
    # state outputted by drive policy net: ([steer], [motor], [speed], [pitch], [yaw])
    state_output = drive_policy_net.forward(dict_input)
    #dictionary representing the state outputted by the drive policy net
    dict_output = {'steer': state_output[0][0], 'motor': state_output[1][0], 'speed': state_output[2][0],
                   'pitch': state_output[3][0], 'yaw': state_output[4][0]}
    # To solve "Not An Element Of Tensor Graph" bug
    # https://justttry.github.io/justttry.github.io/not-an-element-of-Tensor-graph/
    with graph.as_default():
        if args.isTraining:
            if not is_replay:
                if is_first:
                    is_first = False
                    # predict the steering angle for the image
                    rlAgent.memory._step += 1

                    [steer, motor] = rlAgent.getBestAction(dict_output)
                    rlAgent.memory._st = dict_output
                    rlAgent.memory._at = {'steer':steer, 'speed':motor} #Action is delta steer, delta motor

                    # convert the motor value range back to -1000 to 1000 
                    motor = motor * 1000
            
                    #return steer, motor
                    #Driving policy outputs delta steer, delta motor, so add to current steer and motor to get
                    return curr_steer + steer, curr_motor + motor
                else:                    
                    if not is_off_road:
                        # predict the steering angle for the image
                        [Q, steer, motor] = rlAgent.getQ_value_and_best_action(dict_output)
                        #expected imu is a 3x1 vector holding the speed pitch and yaw of state_output
                        expected_imu = [state_output[2][0], state_output[3][0], state_output[4][0]]
                        #Find euclidean distance between current speed, pitch and yaw and the expected imu outputs.
                        difference = euclidean(expected_imu, [speed, pitch, yaw])
                        #Calculate reward. r = a_1(tanh(ln(d_l + d_r))) + a_2(tanh(ln(d_s)))
                        immediate_reward = alpha1 * math.tanh(math.log(dist_left + dist_right, math.e)) + alpha2 * math.tanh(math.log(difference, math.e))
                        reward = immediate_reward + args.gamma * Q
                        rlAgent.memory._r_st1 = reward
                        rlAgent.memory.push()
                        
                        if rlAgent.memory.is_full():
                            is_replay = True
                        else:
                            rlAgent.memory._step += 1
                            rlAgent.memory._st = dict_output
                            rlAgent.memory._at = {'steer':steer, 'motor':motor}
                        
                        # convert the motor value range back to -1000 to 1000 
                        motor = motor * 1000
            
                        #return steer, motor
                        # Driving policy outputs delta steer, delta motor, so add to current steer and motor to get
                        return curr_steer + steer, curr_motor + motor
                    else:                        
                        # off road, reward = -1 and the system goes to the end state
                        reward = -1
                        rlAgent.memory._r_st1 = -1
                        rlAgent.memory.push()
                        # reset system state
                        is_first = True
                        is_off_road = False
                        
                        if rlAgent.memory.is_full():
                            is_replay = True
                            
                        return 0, 0
                        #return curr_steer, curr_motor
            else:             
                # exprience replay 
                rlAgent.experience_replay()   
                # reset system state
                is_first = True
                is_off_road = False
                is_replay = False
                return 0, 0
                
                
                
        else:
            # augment speed metainput
            speed_arg = utils.format_metadata(speed, pitch, yaw)

            # predict the steering angle for the image
            [steer, motor] = rlAgent.getBestAction(dict_output)

            # convert the motor value range back to -1000 to 1000
            motor = motor * 1000
            
            #return steer, motor
            # Driving policy outputs delta steer, delta motor, so add to current steer and motor to get
            return curr_steer + steer, curr_motor + motor
                

#registering event handler for the server
@sio.on('offRoad')
def offRoad(sid, data):
    global is_off_road
    is_off_road = True
    print("offroad! Reset system...")
    rl_drive(None, None)
    send_offRoad()

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        # global speed
        speed = float(data["speed"])
        #current pitch of car
        pitch = float(data["pitch"])
        #current yaw angle of car
        yaw = float(data["yaw"])
        #distance of car to edge of left road
        dist_left = float(data["dist_left"])
        #distance of car to edge of right road
        dist_right = float(data["dist_right"])


        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        try:
            image = np.asarray(image)       # from PIL image to numpy array
            image = utils.preprocess(image)       # apply the preprocessing
            image = np.array([image])       # the model expects 4D array   

            steer, motor = rl_drive(image, steering_angle, throttle, speed, pitch, yaw, dist_left, dist_right)
            print('{} {} {}'.format(steer, motor, speed))
            send_control(steer, motor)            
        except Exception as e:
            print(e)
    else:        
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)        
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)
    
def send_offRoad():
    sio.emit("offRoad", data={}, skip_sid=True)    
    
    
def main():
    global sio, app, graph, args, rlAgent
    parser = argparse.ArgumentParser(description='Reinforcement Learing Driving Program.')    
    parser.add_argument('-log', help='log directory',           dest='log_name',          type=str,   default='RL')
    parser.add_argument('-rl',  help='learning mode',           dest='isTraining',        type=s2b,   default='true')
    parser.add_argument('-rpl',  help='experience replay time', dest='replay_time',       type=int,   default=600)
    
    parser.add_argument('-m', help='pred model directory',      dest='model',             type=str,   default=None)
    parser.add_argument('-t', help='test size fraction',        dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',      dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',          dest='nb_epoch',          type=int,   default=100)
    parser.add_argument('-s', help='samples per epoch',         dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',                dest='batch_size',        type=int,   default=20)
    parser.add_argument('-o', help='save best models only',     dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',             dest='learning_rate',     type=float, default=0.01)
    parser.add_argument('-g', help='gamma',                     dest='gamma',             type=float, default=0.4)
    parser.add_argument('-e', help='random explore',            dest='explore',           type=float, default=0.0001)
    parser.add_argument('-mm', help='momentum',                 dest='momentum',          type=float, default=0.8)
    parser.add_argument('-dc', help='decay',                    dest='decay',             type=float, default=1.0e-6)
    args = parser.parse_args()
    
    try:        
        rlAgent = RLAgent(args)
        #load pre-trained weight
        print "model path:{}".format(args.model)                    
        rlAgent.drive_net.model_init(args.model)              
        
        # To solve "Not An Element Of Tensor Graph" bug
        # https://justttry.github.io/justttry.github.io/not-an-element-of-Tensor-graph/
        graph = tf.get_default_graph()

        # wrap Flask application with engineio's middleware
        app = socketio.Middleware(sio, app)

        # deploy as an eventlet WSGI server
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    except KeyboardInterrupt as e:
        print(e)
    

if __name__ == '__main__':
    main()
    
    
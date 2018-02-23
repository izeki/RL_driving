#parsing command line arguments
import argparse
#decoding camera images
import base64
#for frametimestamp saving
from datetime import datetime
#reading and writing files
import os
#high level file operations
import shutil
#matrix math
import numpy as np
#real-time server
import socketio
#concurrent networking 
import eventlet
#web server gateway interface
import eventlet.wsgi
#threading
import threading
#Queue
import Queue
#image manipulation
from PIL import Image
#web framework
from flask import Flask
#input output
from io import BytesIO

# To solve "Not An Element Of Tensor Graph" bug
# https://justttry.github.io/justttry.github.io/not-an-element-of-Tensor-graph/
import tensorflow as tf
graph = None

from utils import INPUT_SHAPE, format_metadata
from nets import Net
#from nets.SqueezeNet import SqueezeNet
#from nets.SqueezeNet_LSTM import SqueezeLSTMNet
from nets.SqueezeNet_speed import SqueezeSpeedNet

#helper class
import utils

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
#init our model and image array as empty
model = None
prev_image_array = None

#set min/max speed for our autonomous car
MAX_SPEED = 60
MIN_SPEED = 10


#and a speed limit
speed_limit = MAX_SPEED
speed = 0

#image queue
image_queue = Queue.Queue()
cmd_steer_queue = Queue.Queue()
cmd_throttle_queue = Queue.Queue()
speed_queue = Queue.Queue()
is_stop = False

def end_to_end_drive_thread():
    global image_list, is_stop, graph
    while is_stop != True:        
        if not image_queue.empty():
            # SqueezeSpeedNet
            speed = speed_queue.get()
            # augment speed metainput
            speed_arg = format_metadata(speed)
            image = image_queue.get()   
            
            # predict the steering angle for the image
            
            # To solve "Not An Element Of Tensor Graph" bug
            # https://justttry.github.io/justttry.github.io/not-an-element-of-Tensor-graph/
            with graph.as_default():
                """
                # SqueezeNet
                ([steering_angle], [motor])  = drive_net.forward({'IMG_input': image})
                """
                # SqueezeSpeedNet                
                ([steering_angle], [motor], _, __, ___)  = drive_net.forward({'IMG_input': image,
                                                                  'speed_input': speed_arg})
                
                """
                # lower the throttle as the speed increases
                # if the speed is above the current speed limit, we are on a downhill.
                # make sure we slow down first and then go back to the original max speed.                
                global speed_limit, speed
                if speed > speed_limit:
                    speed_limit = MIN_SPEED  # slow down
                else:
                    speed_limit = MAX_SPEED
                throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2
                """
                cmd_steer_queue.put(steering_angle)
                # convert the motor value range back to -1000 to 1000 
                motor = motor * 1000
                cmd_throttle_queue.put(motor)
                
                

#initialize thread
end2end_thread = threading.Thread(target=end_to_end_drive_thread)            
 
    
    
#registering event handler for the server
@sio.on('offRoad')
def offRoad(sid, data):
    global image_queue, is_stop, cmd_steer_queue, cmd_throttle_queue, speed_queue
    print("offroad! Reset system...")
    image_queue = Queue.Queue()
    cmd_steer_queue = Queue.Queue()
    cmd_throttle_queue = Queue.Queue()
    speed_queue = Queue.Queue()
    send_offRoad()

@sio.on('telemetry')
def telemetry(sid, data):
    global image_queue, is_stop, cmd_steer_queue, cmd_throttle_queue, speed_queue
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        # global speed
        speed = float(data["speed"])        
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        try:
            image = np.asarray(image)       # from PIL image to numpy array
            image = utils.preprocess(image) # apply the preprocessing
            image = np.array([image])       # the model expects 4D array   

            """
            # predict the steering angle for the image
            steering_angle = float(model.predict(image, batch_size=1))
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit, speed
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
            """

            #print("push image")
            image_queue.put(image) 
            speed_queue.put(speed)
            send_cmd = False
            info = dict()
            for var, queue in (('cmd_steer', cmd_steer_queue), 
                               ('cmd_throttle', cmd_throttle_queue)):
                if not queue.empty():
                    send_cmd = True
                    info[var] = queue.get()
            if send_cmd:
                print('{} {}'.format(info['cmd_steer'], info['cmd_throttle']))
                send_control(info['cmd_steer'], info['cmd_throttle'])
            else:
                send_control(0, 0)
        except Exception as e:
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))        
    else:
        image_queue = Queue.Queue()
        cmd_steer_queue = Queue.Queue()
        cmd_throttle_queue = Queue.Queue()
        speed_queue = Queue.Queue()
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    global image_queue, is_stop, cmd_steer_queue, cmd_throttle_queue, speed_queue
    print("connect ", sid)
    is_stop = False;
    end2end_thread = threading.Thread(target=end_to_end_drive_thread) 
    image_queue = Queue.Queue()
    cmd_steer_queue = Queue.Queue()
    cmd_throttle_queue = Queue.Queue()
    speed_queue = Queue.Queue()
    # start end2end drive thread
    print("start end2end drive thread ...")
    end2end_thread.start()    
    send_control(0, 0)

"""    
@sio.on('disconnect')    
def disconnect(sid):
    global is_stop
    print("disconnect ", sid)
    is_stop = True
    end2end_thread.join()
"""


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


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Remote Driving')
        parser.add_argument(
            'model',
            type=str,
            help='Path to model h5 file. Model should be on the same path.'
        )
        parser.add_argument(
            'image_folder',
            type=str,
            nargs='?',
            default='',
            help='Path to image folder. This is where the images from the run will be saved.'
        )
        args = parser.parse_args()

        #load model
        print "model path:{}".format(args.model)        
        #drive_net = SqueezeNet(INPUT_SHAPE)   
        #drive_net = SqueezeLSTMNet(INPUT_SHAPE)   
        drive_net = SqueezeSpeedNet(INPUT_SHAPE)   
        
        drive_net.model_init(args.model)        
        #drive_net.model_init()        
        
        # To solve "Not An Element Of Tensor Graph" bug
        # https://justttry.github.io/justttry.github.io/not-an-element-of-Tensor-graph/
        graph = tf.get_default_graph()

        if args.image_folder != '':
            print("Creating image folder at {}".format(args.image_folder))
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                shutil.rmtree(args.image_folder)
                os.makedirs(args.image_folder)
            print("RECORDING THIS RUN ...")
        else:
            print("NOT RECORDING THIS RUN ...")        

        # wrap Flask application with engineio's middleware
        app = socketio.Middleware(sio, app)

        # deploy as an eventlet WSGI server
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    except KeyboardInterrupt as e:
        print(e)
        is_stop = True
        end2end_thread.join()
    except Exception as e:
        print(e)
        is_stop = True
        end2end_thread.join()
    

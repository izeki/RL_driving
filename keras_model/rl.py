#parsing command line arguments
import argparse
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
# RL class
from rl_agent import RLAgent
# To solve "Not An Element Of Tensor Graph" bug
# https://justttry.github.io/justttry.github.io/not-an-element-of-Tensor-graph/
import tensorflow as tf
graph = None

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)

#set min/max speed for our autonomous car
MAX_SPEED = 60
MIN_SPEED = 10

#and a speed limit
speed_limit = MAX_SPEED

#image queue
image_queue = Queue.Queue()
cmd_steer_queue = Queue.Queue()
cmd_throttle_queue = Queue.Queue()
speed_queue = Queue.Queue()
is_stop = False

#for command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'





def main():
    parser = argparse.ArgumentParser(description='Reinforcement Learing Driving Program.')
    
    parser.add_argument('model',
                        type=str,
                        help='Path to model h5 file. Model should be on the same path.')        
    
    parser.add_argument('-log', help='log directory',           dest='log_name',          type=str,   default='RL')
    parser.add_argument('-rl',  help='learning mode',           dest='isTraining',        type=s2b,   default='true')
    parser.add_argument('-rpl',  help='experience replay time', dest='replay_time',       type=int,   default=108000)
    
    parser.add_argument('-m', help='pred model directory',      dest='model_dir',         type=str,   default='')
    parser.add_argument('-t', help='test size fraction',        dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',      dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',          dest='nb_epoch',          type=int,   default=100)
    parser.add_argument('-s', help='samples per epoch',         dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',                dest='batch_size',        type=int,   default=20)
    parser.add_argument('-o', help='save best models only',     dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',             dest='learning_rate',     type=float, default=0.01)
    parser.add_argument('-mm', help='momentum',                 dest='momentum',          type=float, default=0.8)
    parser.add_argument('-dc', help='decay',                    dest='decay',             type=float, default=1.0e-6)
    args = parser.parse_args()

    try:
        #load model
        print "model path:{}".format(args.model)        
        
        rlAgent = RLAgent(args)
        
        #drive_net = SqueezeNet(INPUT_SHAPE)   
        #drive_net = SqueezeLSTMNet(INPUT_SHAPE)   
        drive_net = SqueezeSpeedNet(INPUT_SHAPE)   
        
        drive_net.model_init(args.model)        
        #drive_net.model_init()        
        
        # To solve "Not An Element Of Tensor Graph" bug
        # https://justttry.github.io/justttry.github.io/not-an-element-of-Tensor-graph/
        graph = tf.get_default_graph()

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

if __name__ == '__main__':
    main()
    
    
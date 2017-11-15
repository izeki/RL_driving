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

import time

#initialize our server
sio = socketio.Server()
sio_data = socketio.Server()
#our flask (web) app
app = Flask(__name__)
app_data = Flask('data')

@sio.on('telemetry')
def telemetry(sid, data):
    
    if data:
        print('telemetry with data')    
        send_control(0,0)
    else:
        print('telemetry, time {:f}'.format(time.time()))
        sio.emit('manual', data={}, skip_sid=True)
    
    
@sio.on('offRoad')
def offRoad(sid, data):
    print('offRoad')
    
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0,0)
    
    
@sio_data.on('connect') 
def connect_data(sid, environ):
    print("connect_data", sid)
    send_data()
    
@sio_data.on('data')     
def onData(sid, data):
    print("onData, time {:f}".format(time.time()))
    send_data()

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)    
    
def send_data():
    sio_data.emit('data', data={}, skip_sid=True)    
    
    
def control_socket_thread():    
    eventlet.wsgi.server(eventlet.listen((args.address, args.port)), app)        
    
def data_socket_thread():    
    eventlet.wsgi.server(eventlet.listen((args.address, 8910)), app_data)
    
    
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Socket Tester')
        parser.add_argument(
            'address',
            type=str,
            nargs='?',
            default='',
            help='Binding address for socket io.'
        )
        parser.add_argument(
            'port',
            type=int,
            nargs='?',
            default=4567,
            help='Binding address for socket io.'
        )
        
        args = parser.parse_args()
        
        # wrap Flask application with engineio's middleware
        app = socketio.Middleware(sio, app)
        
        app_data = socketio.Middleware(sio_data, app_data)

        # deploy as an eventlet WSGI server
        #eventlet.wsgi.server(eventlet.listen((args.address, args.port)), app)
        control_thread = threading.Thread(target=control_socket_thread)
        control_thread.start()
        
        #eventlet.wsgi.server(eventlet.listen((args.address, 8910)), app_data)
        
        # Solve greenlet.error: cannot switch to a different thread
        # https://github.com/miguelgrinberg/Flask-SocketIO/issues/65
        from gevent import monkey
        monkey.patch_all()
        
        data_thread = threading.Thread(target=data_socket_thread) 
        data_thread.start()
        
    except KeyboardInterrupt as e:
        control_thread.join()
        data_thread.join()
        print(e)       
    except Exception as e:
        print(e)
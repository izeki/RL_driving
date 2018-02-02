# Basic abstract Net class.
from keras import backend as K
from keras import optimizers

import numpy as np
from utils import INPUT_SHAPE

class Net(object):
    def __init__(self, input_shape=INPUT_SHAPE):
        self.net = None
        self.input_shape=input_shape
        self.N_CHANNEL = 3
        self.N_FRAMES = 1
        self.N_STEPS = 1
    
    def model_init(self, weight_file_path=None):
        def load_model_weight(model, weight_file_path):    
            model.load_weights(weight_file_path, by_name=True)            
            return model
        
        model = self._get_model()
        
        if weight_file_path != None:
            model = load_model_weight(model, weight_file_path)
        
        model.compile(
                loss = 'mean_squared_error', 
                optimizer = optimizers.SGD(
                                    lr = 0.01,
                                    momentum = 0.8, 
                                    decay = 1.0e-6,
                                    nesterov = True),
                metrics=['accuracy'])
        self.net = model
        
        
    def model_compile(self, 
                      learning_rate,
                      momentum,
                      decay,
                      nesterov = True):
        model = self.net
        
        model.compile(
                loss = 'mean_squared_error', 
                optimizer = optimizers.SGD(
                                    lr = learning_rate,
                                    momentum = momentum, 
                                    decay = decay,
                                    nesterov = nesterov),
                metrics=['accuracy'])
        self.net = model
        
        
    def _get_model(self):
        raise NotImplementedError
        
    def get_layer_output(self, model_input, training_flag = True):
        raise NotImplementedError
        
    def forward_backward(self, model_input, target_output):
        raise NotImplementedError
        
    def forward(self, model_input):
        raise NotImplementedError
    
    def save(self, model_path):
        self.net.save(model_path)
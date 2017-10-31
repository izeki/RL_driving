from keras.models import Model
from keras.layers import Input, BatchNormalization, AveragePooling2D, Conv2D, \
                         MaxPooling2D, Dense, ZeroPadding2D, Flatten, concatenate, \
                         LSTM, ConvLSTM2D
from keras import optimizers
from keras import regularizers
from keras.layers.core import Lambda, Dropout, Reshape, Dense
from keras.layers import Activation, Merge
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers, constraints
from Net import Net
from utils import INPUT_SHAPE
import numpy as np

def fire(name, squeeze_planes, expand1x1_planes, expand3x3_planes, **kwargs):
    def f(input):
        squeeze1x1 = Conv2D(filters=squeeze_planes,
                            kernel_size=1, 
                            padding='valid', 
                            activation='relu',                             
                            name='squeeze1x1_'+name)(input)
        expand1x1 = Conv2D(filters=expand1x1_planes,
                           kernel_size=1, 
                           padding='valid', 
                           activation='relu',                            
                           name='expand1x1_'+name)(squeeze1x1)
        expand3x3 = Conv2D(filters=expand3x3_planes, 
                           kernel_size=3, 
                           padding='valid', 
                           activation='relu',                            
                           name='expand3x3_'+name)(squeeze1x1)
        expand3x3 = ZeroPadding2D(padding=(1, 1))(expand3x3)
        return concatenate([expand1x1, expand3x3], axis=3, name='concat'+name)
    return f    

    
class SqueezeLSTMNet(Net):
    def __init__(self, input_shape = INPUT_SHAPE):
        super(SqueezeLSTMNet, self).__init__(input_shape)
        
    
    def _get_model(self):
        IMG_data = Input(shape=self.input_shape, name='IMG_input')        
        IMG_data_norm = Lambda(lambda x: x/127.5-1.0, input_shape=self.input_shape)(IMG_data)
        
        
        IMG_data_pool1 = AveragePooling2D(pool_size=(2, 2),
                                          strides=(2,2),
                                          padding='valid',                                          
                                          name='IMG_data_pool1')(IMG_data_norm)
        IMG_data_pool2 = AveragePooling2D(pool_size=(2, 2),
                                          strides=(2,2),
                                          padding='valid',                                          
                                          name='IMG_data_pool2')(IMG_data_pool1)
        conv1 = Conv2D(filters=64,
                       kernel_size=2,
                       strides=(2,2),
                       padding='valid',
                       activation='relu',                       
                       name='conv1')(IMG_data_pool2)
        conv1_pool = MaxPooling2D(pool_size=(2, 2),
                                  strides=(2,2),
                                  padding='valid', 
                                  name='conv1_pool')(conv1)

        fire1 = fire('1', 16, 64, 64)(conv1_pool)
        fire2 = fire('2', 16, 64, 64)(fire1)
        fire_pool1 = MaxPooling2D(pool_size=(3, 3),
                                  strides=(2,2),
                                  padding='valid',                                  
                                  name='fire_pool1')(fire2)        

        fire3 = fire('3',32, 128, 128)(fire_pool1)
        fire4 = fire('4',32, 128, 128)(fire3)
        fire_pool2 = MaxPooling2D(pool_size=(3, 3),
                                  strides=(2,2),
                                  padding='valid',                                  
                                  name='fire_pool2')(fire4)
        fire5 = fire('5',48, 192, 192)(fire_pool2)
        fire6 = fire('6',48, 192, 192)(fire5)
        fire7 = fire('7',64, 256, 256)(fire6)
        fire8 = fire('8',64, 256, 256)(fire7)

        drop1 = Dropout(rate=0.5, name='drop1')(fire8)        
        
        
        conv2 = Conv2D(filters=2 * self.N_STEPS,
                       kernel_size=1, 
                       padding='valid',                        
                       name='conv2')(drop1)
        
        avg_pool1 = AveragePooling2D(pool_size=(4, 4),
                                     strides=(6,6),
                                     padding='valid',                                      
                                     name='avg_pool1')(conv2)
        
        
        #out = Flatten(name='out_step')(avg_pool1)
        out_step = Reshape((1,2), name='Reshape1')(avg_pool1)        
        
        
        LSTM1 = LSTM(units=4, 
                   use_bias=True, 
                   unit_forget_bias=True,
                   dropout=0.5,
                   recurrent_dropout=0.5,
                   name='LSTM1')(out_step)
        
        Dense1 = Dense(4, name='Dense1')(LSTM1)
        
        out = Dense(2, name='out2')(Dense1)
        
        
        model = Model(inputs=IMG_data, outputs=out) 
        
        return model
    
    def get_layer_output(self, model_input, training_flag = True):        
        get_outputs = K.function([self.net.layers[0].input, K.learning_phase()],
                                 [self.net.layers[52].output])
        layer_outputs = get_outputs([model_input['IMG_input'], training_flag])[0]
        return layer_outputs 
    
    def forward_backward(self, model_input, target_output):
        [loss, accuracy] = self.net.train_on_batch({'IMG_input':model_input['IMG_input']},
                                                   {'out2': target_output['steer_motor_target_data']})
        return loss
        
    def forward(self, model_input):
        prediction = self.net.predict_on_batch({'IMG_input':model_input['IMG_input']})
        ai_steer = (prediction[:,0])
        ai_motor = (prediction[:,1])
        return ai_steer.tolist(), ai_motor.tolist()


def unit_test():
    test_net = SqueezeLSTMNet((376, 672, 3))
    test_net.model_init()
    test_net.net.summary()
    a = test_net.forward({'IMG_input': np.random.rand(1, 376, 672, 3)})
    
    print(a)


unit_test()

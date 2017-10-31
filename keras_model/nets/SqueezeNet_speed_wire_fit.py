from keras.models import Model
from keras.layers import Input, BatchNormalization, AveragePooling2D, Conv2D, \
                         MaxPooling2D, Dense, ZeroPadding2D, Flatten, concatenate
from keras import optimizers
from keras import regularizers
from keras.layers.core import Lambda, Dropout, Reshape
from keras.layers import Activation, Merge
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers, constraints
from Net import Net
import tensorflow as tf
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

    
class SqueezeSpeedWireFitNet(Net):
    def __init__(self, input_shape = INPUT_SHAPE):
        super(SqueezeSpeedWireFitNet, self).__init__(input_shape)
        
    
    def _get_model(self):
        IMG_data = Input(shape=self.input_shape, name='IMG_input')        
        IMG_data_norm = Lambda(lambda x: x/127.5-1.0, input_shape=self.input_shape)(IMG_data)
        
        metadata = Input(shape=(11, 20, 1), name='speed_input')
        
        
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

        fire_pool1_metadata_concat = concatenate([fire_pool1, metadata], axis=3, name='fire_pool1_metadata_concat') 
        
        fire3 = fire('3',32, 128, 128)(fire_pool1_metadata_concat)
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

        flat1 = Flatten(name='flat1')(avg_pool1)
        
        out = Dense(units=24, name='out_q_a')(flat1)
        
        model = Model(inputs=[IMG_data, metadata], outputs=out) 
        
        return model
    
    def model_init(self, weight_file_path=None):
        def load_model_weight(model, weight_file_path):    
            model.load_weights(weight_file_path, by_name=True)            
            return model
        
        model = self._get_model()
        
        if weight_file_path != None:
            model = load_model_weight(model, weight_file_path)
        
        model.compile(
                loss = wire_fit_loss, 
                optimizer = optimizers.SGD(
                                    lr = 0.01,
                                    momentum = 0.8, 
                                    decay = 1.0e-6,
                                    nesterov = True),
                metrics=['loss', 'accuracy', summary])
        self.net = model
    
    def model_compile(self, 
                      learning_rate,
                      momentum,
                      decay,
                      nesterov = True):
        model = self.net
        
        model.compile(
                loss = wire_fit_loss, 
                optimizer = optimizers.SGD(
                                    lr = learning_rate,
                                    momentum = momentum, 
                                    decay = decay,
                                    nesterov = nesterov),
                metrics=['loss', 'accuracy', summary])
        self.net = model
    
    def get_layer_output(self, model_input, training_flag = True):        
        get_outputs = K.function([self.net.layers[0].input, 
                                  self.net.layers[16].input, K.learning_phase()],
                                 [self.net.layers[52].output])
        layer_outputs = get_outputs([model_input['IMG_input'], training_flag])[0]
        return layer_outputs 
    
    def forward_backward(self, model_input, target_output):
        losses = self.net.train_on_batch({'IMG_input':model_input['IMG_input'],
                                          'speed_input':model_input['speed_input']},
                                         {'out_q_a': target_output['q_s_a']})
        return dict(zip(self.net.metrics_names, losses))
        
    def forward(self, model_input):
        q_index = np.array([0,3,6,9,12,15,18,21])
        a_index = np.array([[1,2], 
                            [4,5], 
                            [7,8], 
                            [10,11], 
                            [13,14], 
                            [16,17], 
                            [19,20], 
                            [22,23]])
        prediction = self.net.predict_on_batch({'IMG_input':model_input['IMG_input'],
                                                'speed_input':model_input['speed_input']})
        q = prediction[0][q_index]
        a = prediction[0][a_index]
        return q, a

# wire_fit learning
#
# y_true : r + gamma * max_a(Q(s,a))          
# 
# y_pred : <q_i, a_i>, q_i is the value function approximator, a_i is the policy approximator 
#
#                wsum(s,a)
# Q(s,a) = lim  ----------
#          e->0  norm(s,a)
#
#              n    q_i(s)
# wsum(s,a) = sum ---------
#             i=0  d_i(s,a)
#
#              n      1
# norm(s,a) = sum --------- 
#             i=0  d_i(s,a)
#
# d_i(s,a) = |a-a_i(s)|^2 + c_i * (q_max(s) - q_i(s)) + e 
#
#  dQ          norm(s,a)*(d_k(s,a)+q_k*c_k) - wsum(s,a) * c_k
# ---- = lim  -----------------------------------------------
# dq_k   e->0           (norm(s,a) * d_k(s,a))^2
#
#  dQ          (wsum(s,a) - norm(s,a)* q_k)* 2 * (a_k - a)
# ---- = lim  -----------------------------------------------
# da_k   e->0           (norm(s,a) * d_k(s,a))^2
#
def wire_fit_loss(y_true, y_pred):
    q_index = [0,3,6,9,12,15,18,21]
    a_index = [[1,2], 
               [4,5], 
               [7,8], 
               [10,11], 
               [13,14], 
               [16,17], 
               [19,20], 
               [22,23]]
    x = K.variable(m)
    lr = 0.001
    c = -0.001
    e = 1e-08
    q_idx = K.variable(q_index, dtype='int32')
    a_idx = K.variable(a_index, dtype='int32')
    q = K.gather(x, q_idx)
    a = K.gather(x, a_idx)
    q_max = K.max(q)
    q_max_arg = K.argmax(q)
    a_arg = K.gather(a, q_max_arg)
    d = K.sqrt(K.sum(K.square((a-a_arg)), 1)) + c * (q - q_max) + e
    wsum = K.sum(q / d)
    norm = K.sum(1/d)
    Q = wsum/norm
    dq = lr * (y_true - Q) * (norm * (d + c * q) - wsum * c) / K.square(norm * d)
    da = lr * (y_true - Q) *((wsum - norm * K.transpose(K.stack([q,q]))) * 2 * (a - a_arg)) / K.square(norm * K.transpose(K.stack([d,d])))
    loss_q = K.mean(K.sum(K.square(dq)))
    loss_a = K.mean(K.sum(K.square(da)))
    loss = (loss_q + loss_a)/2
    # log state value and corresponding loss
    tf.summary.scalar("loss_state_value", K.sum(loss_q))
    tf.summary.scalar("loss_action", K.sum(loss_a))
    tf.summary.scalar("state_value", K.sum(q_max))
    
    
    return loss    

# pass a custom metric function to model's compile() call 
# which returns aggregated summary tensor.
# https://groups.google.com/forum/#!topic/keras-users/rEJ1xYqD3AM
def summary(y_true, y_pred):
    return tf.summary.merge_all()    

def unit_test():
    test_net = SqueezeSpeedWireFitNet((376, 672, 3))
    test_net.model_init()
    test_net.net.summary()
    a = test_net.forward({'IMG_input': np.random.rand(1, 376, 672, 3),
                          'speed_input': np.random.rand(1, 11, 20, 1)})
    
    print(a)


unit_test()

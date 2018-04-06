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
from .Net import Net
import numpy as np
import tensorflow as tf


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 376, 672, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def fire(name, squeeze_planes, expand1x1_planes, expand3x3_planes, **kwargs):
    def f(input):
        squeeze1x1 = Conv2D(filters=squeeze_planes,
                            kernel_size=1,
                            padding='valid',
                            activation='relu',
                            name='squeeze1x1_' + name)(input)
        expand1x1 = Conv2D(filters=expand1x1_planes,
                           kernel_size=1,
                           padding='valid',
                           activation='relu',
                           name='expand1x1_' + name)(squeeze1x1)
        expand3x3 = Conv2D(filters=expand3x3_planes,
                           kernel_size=3,
                           padding='valid',
                           activation='relu',
                           name='expand3x3_' + name)(squeeze1x1)
        expand3x3 = ZeroPadding2D(padding=(1, 1))(expand3x3)
        return concatenate([expand1x1, expand3x3], axis=3, name='concat' + name)

    return f


class DrivingPolicyNet(Net):
    def __init__(self, input_shape=INPUT_SHAPE):
        super(DrivingPolicyNet, self).__init__(input_shape)

    def _get_model(self):
        IMG_data = Input(shape=self.input_shape, name='IMG_input')
        IMG_data_norm = Lambda(lambda x: x / 127.5 - 1.0, input_shape=self.input_shape)(IMG_data)


        metadata = Input(shape=(11, 20, 3), name='IMU_input') #3 inputs from IMU: speed, pitch, yaw

        IMG_data_pool1 = AveragePooling2D(pool_size=(2, 2),
                                          strides=(2, 2),
                                          padding='valid',
                                          name='IMG_data_pool1')(IMG_data_norm)

        IMG_data_pool2 = AveragePooling2D(pool_size=(2, 2),
                                          strides=(2, 2),
                                          padding='valid',
                                          name='IMG_data_pool2')(IMG_data_pool1)
        conv1 = Conv2D(filters=64,
                       kernel_size=2,
                       strides=(2, 2),
                       padding='valid',
                       activation='relu',
                       name='conv1')(IMG_data_pool2)

        conv1_pool = MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2),
                                  padding='valid',
                                  name='conv1_pool')(conv1)

        fire1 = fire('1', 16, 64, 64)(conv1_pool)
        fire2 = fire('2', 16, 64, 64)(fire1)
        fire_pool1 = MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='valid',
                                  name='fire_pool1')(fire2)

        fire_pool1_metadata_concat = concatenate([fire_pool1, metadata], axis=3, name='fire_pool1_metadata_concat')

        fire3 = fire('3', 32, 128, 128)(fire_pool1_metadata_concat)
        fire4 = fire('4', 32, 128, 128)(fire3)
        fire_pool2 = MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='valid',
                                  name='fire_pool2')(fire4)
        fire5 = fire('5', 48, 192, 192)(fire_pool2)
        fire6 = fire('6', 48, 192, 192)(fire5)
        fire7 = fire('7', 64, 256, 256)(fire6)
        fire8 = fire('8', 64, 256, 256)(fire7)

        drop1 = Dropout(rate=0.5, name='drop1')(fire8)
        conv2 = Conv2D(filters=5 * self.N_STEPS, #### Changed filters to 5 so that output has 5 things: steer, motor, next_speed, next_pitch, next_yaw ####
                       kernel_size=1,
                       padding='valid',
                       name='conv2')(drop1)

        avg_pool1 = AveragePooling2D(pool_size=(4, 4),
                                     strides=(6, 6),
                                     padding='valid',
                                     name='avg_pool1')(conv2)
        out = Flatten(name='out3')(avg_pool1)
        #
        """
        with tf.name_scope("steer"):
            steer = out[0,:]
            tf.summary.scalar("steer", steer)
        with tf.name_scope("motor"):
            motor = out[1,:]
            tf.summary.scalar("motor", motor)
        with tf.name_scope("speed"):
            speed = out[2, :]
            tf.summary.scalar("speed", speed)
        """
        model = Model(inputs=[IMG_data, metadata], outputs=out)
        
        return model

    def get_layer_output(self, model_input, training_flag=True):
        get_outputs = K.function([self.net.layers[0].input,
                                  self.net.layers[17].input, K.learning_phase()],  #Changed 16 to 17
                                 [self.net.layers[53].output])     #Changed 52 to 53
        layer_outputs = get_outputs([model_input['IMG_input'], model_input['IMU_input'], training_flag])[0]
        return layer_outputs

    def forward_backward(self, model_input, target_output):
        [loss, accuracy] = self.net.train_on_batch({'IMG_input': model_input['IMG_input'],
                                                    'IMU_input': model_input['IMU_input']},
                                                   {'out3': target_output['out3']})
        return loss

    
    def forward(self, model_input, target_output):
        prediction = self.net.predict_on_batch({'IMG_input': model_input['IMG_input'],
                                                'IMU_input': model_input['IMU_input']})

        ai_steer = (prediction[:, 0])
        ai_motor = (prediction[:, 1])
        ai_speed = (prediction[:, 2])
        ai_pitch = (prediction[:, 3])
        ai_yaw = (prediction[:, 4])


        return ai_steer, ai_motor, ai_speed, ai_pitch, ai_yaw
    
def unit_test():
    test_net = DrivingPolicyNet((376, 672, 3))
    test_net.model_init()
    test_net.net.summary()
    input1_img = np.random.rand(1, 376, 672, 3)
    input1_IMU = np.random.rand(1, 11, 20, 3)
    a = test_net.forward({'IMG_input': input1_img, 'IMU_input': input1_IMU})
    b = test_net.get_layer_output({'IMG_input': input1_img, 'IMU_input': input1_IMU})
    print(a)
    #print(b)

#unit_test()
"""
with tf.Session as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/train',
                                         sess.graph)
    test_writer = tf.summary.FileWriter('/test')
"""

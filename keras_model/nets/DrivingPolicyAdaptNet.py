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
from utils import INPUT_SHAPE
import numpy as np

class DrivingPolicyAdaptNet(Net):
    def __init__(self, input_shape=INPUT_SHAPE):
        super(DrivingPolicyAdaptNet, self).__init__(input_shape)

    def _get_model(self):

        input = Input(shape=(5,), name='State_input') #changed dimension to 5 as there are now 5 inputs for state: steer, motor, speed, pitch, yaw

        out = Dense(units=24, name='Action_output')(input)

        model = Model(inputs=[input], outputs=out)

        return model

    def get_layer_output(self, model_input, training_flag=True):
        get_outputs = K.function([self.net.layers[0].input,
                                  K.learning_phase()],  #Changed 16 to 17
                                 [self.net.layers[1].output])     #Changed 52 to 53
        layer_outputs = get_outputs([model_input['State_input'], training_flag])[0]
        return layer_outputs

    def forward_backward(self, model_input, target_output):
        [loss, accuracy] = self.net.train_on_batch({'State_input': model_input['State_input']},
                                                   {'out': target_output['Action_output']})
        return loss

    def forward(self, model_input):

        q_index = np.array([0, 3, 6, 9, 12, 15, 18, 21])
        a_index = np.array([[1, 2],
                            [4, 5],
                            [7, 8],
                            [10, 11],
                            [13, 14],
                            [16, 17],
                            [19, 20],
                            [22, 23]])
        prediction = self.net.predict_on_batch({'State_input': model_input['State_input']})

        q = prediction[0][q_index]
        a = prediction[0][a_index]
        return q, a


def unit_test():
    test_net = DrivingPolicyAdaptNet((5,))
    test_net.model_init()
    test_net.net.summary()
    state_input = np.random.rand(1, 5)
    a = test_net.forward({'State_input': state_input})
    print(a)

unit_test()

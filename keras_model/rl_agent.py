from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import tensorflow as tf
from utils import INPUT_SHAPE, preprocess, format_metadata_RL
from io import BytesIO
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from nets.SqueezeNet_speed_wire_fit import SqueezeSpeedWireFitNet

batch_size = 20 ### DEFAULT TO 20 based on rl.py, can be subject to change #######


class ReplayMemory():
    def __init__(self, size):        
        self.size = size
        self.elements = [] # element: dict{'step':t, 'st':st, 'at':at, 'r_st1':r_st1}
        self._step = 0
        self._st = None # st=dict{'speed':speed, 'pitch':pitch, 'yaw':yaw, 'steer':steer, 'motor':motor}
        self._at = None # at=dict{'steer':steer, 'motor':motor}
        self._st1 = None #st1 = dict{'speed':speed, 'pitch':pitch, 'yaw':yaw, 'steer':steer, 'motor':motor} #Added state of timestep t+1
        self.r_st1 = None # r_st1 = r + gamma * max_a(Q(s_t1,a_t1))
    
    def push(self):
        element = {'step': self._step, 'st':self._st, 'at':self._at, 'r_st1':self._r_st1, 'st1': self._st1}
        self.elements.append(element)
        self._st = None 
        self._at = None 
        self.r_st1 = None 
        self._st1 = None

    def clear(self):
        self.elements = []
        self._st = None # st=dict{'speed':speed, 'pitch':pitch, 'yaw':yaw}
        self._at = None # at=dict{'steer':steer, 'motor':motor}
        self.r_st1 = None # r_st1 = r + gamma * max_a(Q(s_t1,a_t1))
        self._st1 = None
    
    def is_full(self):
        if len(self.elements) >= self.size:
            return True
        else:
            return False
        
class RLAgent():
    def __init__(self, args):        
        self.memory = ReplayMemory(args.replay_time) # 1hour@30fps
        self.drive_net = SqueezeSpeedWireFitNet(INPUT_SHAPE)
        #self.summary_writer = tf.summary.FileWriter("logs/" + args.log_name)   
        self.log_name = args.log_name
        self.samples_per_epoch = args.samples_per_epoch
        self.save_best_only = args.save_best_only
        self.nb_epoch = args.nb_epoch
        self.batch_size = args.batch_size
        self.explore = args.explore
        self.test_size = args.test_size
    
    def getBestAction(self, s):
        q, a = self.drive_net.forward(s)
        q_max_arg = np.argmax(q)
        a_best = a[q_max_arg] + self.explore
        
        return a_best[0], a_best[1] # steer, motor
    
    def getQ_value(self, s, a_best):
        q, a = self.drive_net.forward(s)
        return self.wire_fit(q, a, a_best)
    
    def getQ_value_and_best_action(self, s):
        q, a = self.drive_net.forward(s)
        q_max_arg = np.argmax(q)
        a_best = a[q_max_arg] + self.explore       
        return self.wire_fit(q,a,a_best), a_best[0], a_best[1]
    
    def wire_fit(self, q, a, a_best, c=-0.001, e=1e-08):    
        q_max = np.max(q)
        q_max_arg = np.argmax(q)
        #a_best = a[q_max_arg]
        d = np.sqrt(np.sum(np.square((a-a_best)), 1)) + c * (q - q_max) + e
        wsum = np.sum(q/d)
        norm = np.sum(1/d)
        Q = wsum/norm    
        return Q
    
    def build_model(self, model_path):
        self.drive_net.model_init(model_path)
    
    def experience_replay(self):
        print("start experience replay...")
        m = self.memory.elements
        model = self.drive_net.net
        #generate training/validation set
        X = [{'steer':m[i]['st']['steer'], 'motor':m[i]['st']['motor'], 'speed':m[i]['st']['speed'],
              'pitch':m[i]['st']['pitch'], 'yaw':m[i]['st']['yaw']} for i in range(len(m))] #Modified state of RL to now be steer, motor, speed, pitch and yaw, rather than before where it was only image and speed.
        y = [{'r_st1':m[i]['r_st1'], 'a_best':m[i]['at']} for i in range(len(m))]
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=self.test_size, random_state=0)
        
        #tensorboard unility
        tbCallBack = TensorBoard(log_dir='logs/'+ self.log_name, histogram_freq=0, write_graph=True, write_images=True)
        
        #Reduce learning rate callback
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
        
        #Saves the model after every epoch.
        #quantity to monitor, verbosity i.e logging mode (0 or 1), 
        #if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
        #mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
        # made based on either the maximization or the minimization of the monitored quantity. For val_acc, 
        #this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
        # inferred from the name of the monitored quantity.
        checkpoint = ModelCheckpoint('../models/rl-model-{epoch:03d}.h5',
                                     monitor='val_loss',
                                     verbose=0,
                                     save_best_only=self.save_best_only,
                                     mode='auto')
        
        model.fit_generator(self._mini_batch_generator(X_train, y_train),  ## Added self in front of the generation to resolve reference
                            self.samples_per_epoch/self.batch_size,
                            self.nb_epoch,
                            max_queue_size=1,
                            validation_data=self._mini_batch_generator(X_valid, y_valid),
                            validation_steps=len(X_valid),
                            callbacks=[checkpoint, tbCallBack, reduce_lr],
                            verbose=1)
        
    def _mini_batch_generator(self, st, r_st1_a_best):
        """
        Generate training image give image paths and associated steering angles
        """
        #images = np.empty([batch_size, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
        state = np.empty([batch_size, 11, 20, 5]) #steer, motor, speed, pitch and yaw are the 5 inputs.
        
        outs = np.empty([batch_size, 2]) #Adjusted steer and adjusted motor are the two outputs

        while True:
            i = 0
            for index in np.random.permutation(len(st)):    
                #image = st[index]['img']
                steer = st[index]['steer']
                motor = st[index]['motor']
                speed = st[index]['speed']
                pitch = st[index]['pitch']
                yaw = st[index]['yaw']
                
                target = [r_st1_a_best[index]['r_st1'], r_st1_a_best[index]['a_best']]

                #images[i] = image

                # argument speed meta input
                speed_arg = format_metadata_RL(steer, motor, speed, pitch, yaw)
                state[i] = speed_arg

                outs[i] = target
                i += 1
                if i == self.batch_size:
                    break
            #yield images, steers
            #yield ({'IMG_input': images, 'speed_input': speeds}, {'out_q_a': outs})

            #Now yield state and actions
            yield ({'State_Input':state}, {'Action_Output':outs})
            
            


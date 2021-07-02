from collections import deque
from tensorflow import keras

import os
import random
import numpy as np
import pandas as pd
from pprint import pprint

class DeepQNetwork():
    def __init__(self, states, actions, alpha, gamma, epsilon,epsilon_min, epsilon_decay):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=2500)
        self.alpha = alpha
        self.gamma = gamma
        #Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.loss = []
        
    def build_model(self):



        model= keras.models.Sequential()

        model.add(keras.layers.Convolution2D(32,(4,4),padding="same",input_shape=(7,11,1)))
        model.add(keras.layers.Activation('relu'))


        model.add(keras.layers.Convolution2D(64,(4,4)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.2))


        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(50))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(10))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(4))
        model.add(keras.layers.Activation('softmax'))
        model.compile(loss='mean_squared_error', #Loss function: Mean Squared Error
                      optimizer=keras.optimizers.Adam(lr=self.alpha))

        print(model.summary())



        '''model = keras.Sequential() #linear stack of layers https://keras.io/models/sequential/
        model.add(keras.layers.Dense(11, input_dim=self.nS, activation='relu')) #[Input]
        for i in range(0,layers_num):
            model.add(keras.layers.Dense(layer_neuron_num, activation='relu'))
        model.add(keras.layers.Dense(self.nA, activation='linear')) #[output]
        #   Size has to match the output (different actions)
        #   Linear activation on the last layer
        model.compile(loss='mean_squared_error', #Loss function: Mean Squared Error
                      optimizer=keras.optimizers.Adam(lr=self.alpha)) #Optimaizer: Adam (Feel free to check other options)
'''


        return model

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA) #Explore
        state = state.reshape(1,7,11,1)
        action_vals = self.model.predict(state) #Exploit: Use the NN to predict the correct action from this state
        return np.argmax(action_vals[0])

    def test_action(self, state): #Exploit
        state = state.reshape(1,7,11,1)
        action_vals = self.model.predict(state)
        return np.argmax(action_vals[0])

    def store(self, state, action, reward, nstate, done):
        #Store the experience in memory
        self.memory.append( (state, action, reward, nstate, done) )

    def experience_replay(self, batch_size):
        #Execute the experience replay
        minibatch = random.sample( self.memory, batch_size ) #Randomly sample from memory

        #Convert to numpy for speed by vectorization
        x = []
        y = []
        np_array = np.array(minibatch)
        st = np.zeros((0,self.nS)) #States
        #print(st.shape)
        nst = np.zeros( (0,self.nS) )#Next States
        for i in range(len(np_array)): #Creating the state and next state np arrays
            #print("exp",np_array[i,0],np_array[i,0].shape)
            st = np.append( st, np_array[i,0], axis=0)
            nst = np.append( nst, np_array[i,3], axis=0)
        st_predict = self.model.predict(st.reshape(st.shape[0],7,11,1)) #Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst.reshape(nst.shape[0],7,11,1))
        index = 0
        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            #Predict from state
            nst_action_predict_model = nst_predict[index]
            if done == True: #Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward
            else:   #Non terminal
                target = reward + self.gamma * np.amax(nst_action_predict_model)
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1
        #Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size,self.nS)
        y_reshape = np.array(y)
        epoch_count = 1 #Epochs is the number or iterations
        #print("resh",x_reshape.shape)
        hist = self.model.fit(x_reshape.reshape(x_reshape.shape[0],7,11,1), y_reshape, epochs=epoch_count, verbose=0)
        #Graph Losses
        for i in range(epoch_count):
            self.loss.append( hist.history['loss'][i] )
        #Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
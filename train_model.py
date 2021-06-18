import os
import random
import numpy as np
import pandas as pd
from pprint import pprint
from kaggle_environments import make, evaluate
from gym import spaces
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
import random

from HungryGeeseEnv import *
from DeepQNetwork import *

import math
from multiprocessing import Process

def train(layers_num,layer_neuron_num):

    try:
        os.mkdir("./data/"+str(layers_num)+'-'+str(layer_neuron_num))
        print("-------- Iniciando treino para o modelo com "+str(layers_num)+" Hidden Layers e com "+str(layer_neuron_num)+" Neuronios em cada Layer --------")
    except:
        print("Treino ja Executado anteriormente... Abortando execução")
        return
    env = HungryGeeseGym()

    #Global Variables
    EPISODES = 2000
    TRAIN_END = 0
    #Hyper Parameters
    discount_rate = 0.95 #Gamma
    learning_rate = 0.001 #Alpha
    batch_size = 24 #Size of the batch used in the experience replay

    #Create the agent
    nS = env.observation_space.shape[0]
    nA = env.action_space.n
    try:
        del dqn
    except:
        discount_rate
    dqn = DeepQNetwork(nS, nA, learning_rate, discount_rate, 1, 0.001, (0.222)*(1/(EPISODES/2)),layers_num,layer_neuron_num )
    #Training
    rewards = [] #Store rewards for graphing
    epsilons = [] # Store the Explore/Exploit
    TEST_Episodes = 0
    env.debug = False
    for e in tqdm(range(EPISODES),desc = (str(layers_num)+'-'+str(layer_neuron_num)),leave = False):
        #if(e%int(EPISODES/100) == 0):
        #    print(str(layers_num)+'-'+str(layer_neuron_num)+": "+str(math.floor(100*e/EPISODES)) + "%")
        state = env.reset()
        state = np.reshape(state, [1, nS]) # Resize to store in memory to pass to .predict
        tot_rewards = 0
        for time in range(200): 
            action = dqn.action(state)
            nstate, reward, done, _ = env.step(action)
            nstate = np.reshape(nstate, [1, nS])
            tot_rewards += reward
            dqn.store(state, action, reward, nstate, done) # Resize to store in memory to pass to .predict
            state = nstate
            if done:
                rewards.append(tot_rewards)
                epsilons.append(dqn.epsilon)
                '''print("episode: {}/{}, score: {}, e: {}"
                    .format(e, EPISODES, tot_rewards, dqn.epsilon))'''
                break
            #Experience Replay
            if len(dqn.memory) > batch_size:
                dqn.experience_replay(batch_size)
    
    dqn.model.save('./data/'+str(layers_num)+'-'+str(layer_neuron_num)+'/model')
    df0 = pd.DataFrame()
    df0['Reward'] = np.array(rewards)
    df0.to_csv('./data/'+str(layers_num)+'-'+str(layer_neuron_num)+"/modelData-total.csv")
    med = 0
    y_values = []
    x_values = []
    eps_graph = []
    eps = 0
    aux = 0
    BATCH = 10
    for i in range(0,len(rewards)): 
        med+=rewards[i]
        eps+=epsilons[i]
        if i%(BATCH-1) == 0 and i != 0:
            x_values.append(aux)
            aux+=1
            y_values.append(med/BATCH)
            eps_graph.append(40*eps/BATCH)
            eps = 0
            med = 0

    df = pd.DataFrame()
    #print(x_values)
    df["Score"] = np.array(y_values)
    df['Epslon'] = np.array(eps_graph)
    df["round"] = np.array(x_values)
    fig = px.line(df, x='round', y=['Score','Epslon'])
    fig.show()
    df.to_csv('./data/'+str(layers_num)+'-'+str(layer_neuron_num)+"/modelData.csv")
    fig.write_html('./data/'+str(layers_num)+'-'+str(layer_neuron_num)+"plot.html")

def process_handler():
    arqs = []
    for i in range(1,10):
        for j in range(3,45):
            arqs.append([i,j])
    while True:
        n = int(1000*random.random()%len(arqs))
        train(arqs[n][0],arqs[n][1])
        arqs.remove([arqs[n][0],arqs[n][1]])
        #os.system('cls')


if __name__ == '__main__':
    #freeze_support()
    train(7,49)
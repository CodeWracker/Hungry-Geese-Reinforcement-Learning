from kaggle_environments import make, evaluate
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
from kaggle_environments.envs.hungry_geese.hungry_geese import translate, adjacent_positions, min_distance
from gym import spaces

import os
import random
import numpy as np
import pandas as pd
from pprint import pprint

class HungryGeeseGym:
    def __init__(self, agent2="./submission-ralph-coward.py"):
        self.ks_env = make("hungry_geese", debug=False)
        self.env = self.ks_env.train([None, agent2,agent2,agent2])
        self.rows = self.ks_env.configuration.rows
        self.columns = self.ks_env.configuration.columns
        self.observation_space_size = self.rows * self.columns
        # Learn about spaces here: http://gym.openai.com/docs/#spaces
        # Permitted actions
        # ['NORTH', 'EAST', 'SOUTH', 'WEST']
        actL = []
        for action in Action:
            actL.append(action)
        self.actions = actL
        # Defined Action Space(Must)
        self.action_space = spaces.Discrete(len(self.actions))
        
        self.observation_space = spaces.Box(low=np.zeros(shape=(77,), dtype=int), high=np.zeros(shape=(77,), dtype=int)+3)
        
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 1)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None
        self.last = self.action_space.sample()
    def reset(self):
        self.last = self.action_space.sample()
        self.obs = self.env.reset()
        #print(self.obs)
        return self.get_grid(self.obs)
    
    def opposite(self,action):
        if action == Action.NORTH:
            return Action.SOUTH
        if action == Action.SOUTH:
            return Action.NORTH
        if action == Action.EAST:
            return Action.WEST
        if action == Action.WEST:
            return Action.EAST
        
    def step(self, action):
        action = self.actions[action]
        # Check if agent's move is valid
        is_valid = action != self.last
        self.last = self.opposite(action)
        cont = 0
        for pos in self.obs:
            if pos == 0:
                cont +=1
        reward, done, _ = -15, True, {}
        if is_valid: # Play the move
            self.obs, old_reward, done, _ = self.env.step(action.name)
            #print(self.obs)
            if(done):
                reward = 5
                if(self.obs['geese'][self.obs['index']] == []):
                    reward = -5
            else:
                if(len(self.obs['geese'][self.obs['index']])>cont ):
                    reward = 1 + len(self.obs['geese'][self.obs['index']])
                else:
                    reward = -1/50
        
        return self.get_grid(self.obs), reward, done, _
    
    def get_grid(self,obs):
        mapa = []
        for i in range(0,self.rows):
            mapa.append([])
            for j in range(0,self.columns):
                achou = False
                for food in obs['food']:
                    x,y = row_col(food,self.columns)
                    if(x == i and y == j):
                        mapa[i].append(3)
                        achou = True
                        break
                if(achou):
                    continue
                aux = 0
                for goose in obs['geese']:
                    gs = 2
                    if(aux == obs['index']):
                        gs = 1
                    aux = aux +1
                    for part in goose:
                        x,y = row_col(part,self.columns)
                        if(x == i and y == j):
                            mapa[i].append(gs)
                            achou = True
                            break
                    if(achou):
                        break
                if(achou):
                    continue
                mapa[i].append(0)

        return np.array(mapa).reshape(1,-1)[0]
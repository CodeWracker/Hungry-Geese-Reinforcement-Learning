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
        self.debug = True
        for action in Action:
            actL.append(action)
        self.actions = actL
        
        #print(actL)
        # Defined Action Space(Must)
        self.action_space = spaces.Discrete(len(self.actions))
        
        self.observation_space = spaces.Box(low=np.zeros(shape=(77,), dtype=int), high=np.zeros(shape=(77,), dtype=int)+10)
        
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 1000)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None
        self.last = self.actions[self.action_space.sample()]
    def reset(self):
        self.last = self.actions[self.action_space.sample()]
        
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


        cont = len(self.obs['geese'][self.obs['index']])
        #print(cont)
        reward, done, _ = -50, True, {}
        if is_valid: # Play the move
            self.obs, old_reward, done, _ = self.env.step(action.name)
            #print(self.obs)
            if(done):
                reward = 100 # se ganhou
                if(self.obs['geese'][self.obs['index']] == []):
                    reward = -5 # se perdeu
            else:
                if(len(self.obs['geese'][self.obs['index']])>cont ):
                    reward = 20 # se comeu
                else:
                    if(len(self.obs['geese'][self.obs['index']])<cont ):
                        reward = -10 # se comeu
                        #print("Step: "+str(self.obs['step']) + " /  Perdeu tamanho... " + str(reward)+" Pontos")
                    else:
                        reward = -1 # se não fez nada


                
        grid = self.get_grid(self.obs)
        
        '''sensores = self.sensors(grid)
        if(self.debug):
            print(sensores.shape)
            print(sensores)
            print(grid.reshape(1,-1)[0].shape)'''

        # o obs tem que ser [indice da ultima ação. [sens1], [sens2], [sens3]]
        return grid, reward, done, _
    
    def sensors(self,grid):
       
        if(len(self.obs['geese'][self.obs['index']])==0):
            return np.array([0,0,0,0,0,0,0,0,0,0,0])
        px,py = row_col(self.obs['geese'][self.obs['index']][0],self.columns)
        actL = []
        for action in Action:
            actL.append(action)
        frente = 0
        if(self.last):
            for i in range(0,len(actL)):
                if(actL[i] == self.opposite(self.last)):
                    frente = i
                    break
            actL.remove(self.last)
        else: # Diz que esta indo para o norte e remove o oposto (SUL)
            actL.remove(action.SOUTH)
            frente = 0
       

        direita = frente+1
        if(direita>3):
            direita = direita-4
        esquerda = frente + 3
        if(esquerda>3):
            esquerda = esquerda - 4

        movimentos = [
            [-1,0],     #Norte
            [0,1],      #Leste
            [1,0],      #Sul
            [0,-1],     #Oeste
        ]
        sensor_frente = [0,0] #[distancia,tipo (0:inimigo,1:food)]
        for i in range(0,11):
            px_a,py_a = (px+movimentos[frente][0]*(i+1),py+movimentos[frente][1]*(i+1))
            if(px_a>6):
                px_a = px_a - 7
            if(py_a>10):
                py_a = py_a - 11
            
            if(px_a<0):
                px_a = 7 + px_a
            if(py_a<0):
                py_a = 11 + py_a
            
            if(grid[px_a][py_a] == 2 or grid[px_a][py_a] == 1):
                break
            if(grid[px_a][py_a] == 3):
                sensor_frente[1] = 1
                break
            else:
                sensor_frente[0]+=1
        
        sensor_esquerda = [0,0] #[distancia,tipo (0:inimigo,1:food)]
        for i in range(0,11):
            px_a,py_a = (px+movimentos[esquerda][0]*(i+1),py+movimentos[esquerda][1]*(i+1))
            if(px_a>6):
                px_a = px_a - 7
            if(py_a>10):
                py_a = py_a - 11
            
            if(px_a<0):
                px_a = 7+ px_a
            if(py_a<0):
                py_a = 11 + py_a
            
            if(grid[px_a][py_a] == 2 or grid[px_a][py_a] == 1):
                break
            if(grid[px_a][py_a] == 3):
                sensor_esquerda[1] = 1
                break
            else:
                sensor_esquerda[0]+=1
        
        sensor_direita = [0,0] #[distancia,tipo (0:inimigo,1:food)]
        for i in range(0,11):
            px_a,py_a = (px+movimentos[direita][0]*(i+1),py+movimentos[direita][1]*(i+1))
            if(px_a>6):
                px_a = px_a - 7
            if(py_a>10):
                py_a = py_a - 11
            
            if(px_a<0):
                px_a = 7 + px_a
            if(py_a<0):
                py_a = 11 + py_a
            
            
            if(grid[px_a][py_a] == 2 or grid[px_a][py_a] == 1):
                break
            if(grid[px_a][py_a] == 3):
                sensor_direita[1] = 1
                break
            else:
                sensor_direita[0]+=1
        
        # Verificando as diagonais
        tras = frente + 2
        if(tras>=4):
            tras-=4
        
        px_a,py_a = (px+    movimentos[direita][0]  + movimentos[frente][0] )  ,  (py+     movimentos[direita][1]  + movimentos[frente][1])   
        if(px_a>6):
            px_a = px_a - 7
        if(py_a>10):
            py_a = py_a - 11
        
        if(px_a<0):
            px_a = 7+ px_a
        if(py_a<0):
            py_a = 11 + py_a
        frente_direita = grid[px_a,py_a]

        px_a,py_a =   (px+    movimentos[esquerda][0] + movimentos[frente][0] ) ,   (py+     movimentos[esquerda][1] + movimentos[frente][1])
        if(px_a>6):
            px_a = px_a - 7
        if(py_a>10):
            py_a = py_a - 11
        
        if(px_a<0):
            px_a = 7+ px_a
        if(py_a<0):
            py_a = 11 + py_a
        frente_esquerda = grid[px_a,py_a]

        px_a,py_a =      (px+    movimentos[esquerda][0] + movimentos[tras][0]   ) ,   (py+     movimentos[esquerda][1] + movimentos[tras][1]  )
        if(px_a>6):
            px_a = px_a - 7
        if(py_a>10):
            py_a = py_a - 11
        
        if(px_a<0):
            px_a = 7+ px_a
        if(py_a<0):
            py_a = 11 + py_a
        tras_esqueda = grid[px_a,py_a]


        px_a,py_a=      (px+    movimentos[direita][0]  + movimentos[tras][0]   ) ,   (py+     movimentos[direita][1]  + movimentos[tras][1]  )
        if(px_a>6):
            px_a = px_a - 7
        if(py_a>10):
            py_a = py_a - 11
        
        if(px_a<0):
            px_a = 7+ px_a
        if(py_a<0):
            py_a = 11 + py_a
        tras_direita = grid[px_a,py_a]

        if(self.debug):
            print("x:"+str(px)+" / y:"+str(py))
            print("Frente ("+str(self.actions[frente].name) +"):"+str(sensor_frente) + " / Esquerda ("+str(self.actions[esquerda].name) +"):"+str(sensor_esquerda) + " / Direita ("+str(self.actions[direita].name) +"):"+str(sensor_direita))
            print(grid)
        return np.array([frente,sensor_frente[0],sensor_frente[1],sensor_esquerda[0],sensor_esquerda[1],sensor_direita[0],sensor_direita[1],frente_direita,frente_esquerda,tras_direita,tras_esqueda])

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
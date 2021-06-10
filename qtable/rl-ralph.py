import numpy as np
import pandas as pd
from numpy import genfromtxt
from kaggle_environments import make, evaluate
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col
from kaggle_environments.envs.hungry_geese.hungry_geese import translate, adjacent_positions, min_distance
def select_row(obs):
    i = 0
    cont =0
    for a in obs:
        cont+=1
        i+=cont*a
    return i % 35
def get_grid_from_obs(obs,columns,rows):
        mapa = []
        for i in range(0,rows):
            mapa.append([])
            for j in range(0,columns):
                achou = False
                for food in obs['food']:
                    x,y = row_col(food,columns)
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
                        x,y = row_col(part,columns)
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
def agent(obs,config):
    q_table = genfromtxt('./data/180-0.001/qtable.csv', delimiter=',')
    #print(obs)
    actL = []

    for action in Action:
        actL.append(action)
    state = get_grid_from_obs(obs,config.columns,config.rows)
    #print(state)
    action = np.argmax(q_table[select_row(state),:])
    #pred = model.predict(state.reshape(1,-1))
    #aux = 0
    #for i in range (0,len(pred[0])):
    #    if(pred[0][i]>pred[0][aux]):
    #        aux = i
    return actL[action].name
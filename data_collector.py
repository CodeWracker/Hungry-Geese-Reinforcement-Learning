from pandas.io.parsers import read_csv

import pandas as pd
import numpy as np
import os


'''
df = px.data.gapminder()

fig = px.bar(df, x="continent", y="pop", color="continent",
  animation_frame="year", animation_group="country", range_y=[0,4000000000])
fig.show()'''

diretorio = "./data"
pastas = (os.listdir(diretorio))
df = pd.DataFrame()
Score = []
Neuronios = []
Layers = []
Rounds = []
for pasta in pastas:
    layers,neuronios = pasta.split('-')
    try:
      aux_df = read_csv(diretorio+"/"+pasta+"/modelData.csv")
    except:
      continue
    #print(aux_df)
    for i in range(0,len(aux_df["Score"])):
        Layers.append(layers)
        Neuronios.append(neuronios)
        Score.append(aux_df["Score"][i])
        Rounds.append(i)
    #print("a")


df["Score"] = Score
df["Neuronios"] = Neuronios
df["Layers"] = Layers
df["Round"] = Rounds
print(df.info())
df.to_csv("resultados.csv")
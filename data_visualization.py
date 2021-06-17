import numpy as np
import plotly.express as px
import pandas as pd

from pprint import pprint

df = pd.read_csv("./data/7-49/modelData-total.csv",sep=',')

data = df["Reward"]
scores = []
aux = 0
for i,dt in enumerate(data):
    aux+=dt
    if(i%99 == 0):
        scores.append(aux/100)
        aux = 0
fig = px.line(scores)
fig.show()
fig.write_html("./data/7-49/plot-100.html")
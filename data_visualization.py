%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive
import seaborn as sb
import pandas as pd

from pprint import pprint

df = pd.read_csv("resultados.csv",sep=',')
print(df["Layers"][np.argmax(df["Layers"])])
print(df["Neuronios"][np.argmax(df["Neuronios"])])
X = np.linspace(df["Layers"][np.argmin(df["Layers"])],df["Layers"][np.argmax(df["Layers"])] ,df["Layers"][np.argmax(df["Layers"])])
Y = np.linspace(df["Neuronios"][np.argmin(df["Neuronios"])],df["Neuronios"][np.argmax(df["Neuronios"])],df["Neuronios"][np.argmax(df["Neuronios"])]-2)
minimo = (df["Score"][np.argmin(df["Score"])])
maximo = (df["Score"][np.argmax(df["Score"])])
x,y = np.meshgrid(X,Y)


def f(L,N,R):
    r = []
    for i in range(0,len(L)):
        aux = []
        #print(L[i])
        #print(N[i])
        #print(N[i])
        for j in range(0,len(L[i])):
            #print(L[i][j],N[i][j],R)
            try:
                #print("AAA")
                aux.append(float(df.query('Layers == {} and Neuronios == {} and Round == {} '.format(L[i][j],N[i][0],R))["Score"]))
                #print("AAA")
            except:
                #print("BBB")
                aux.append(-50)
                #print("BBB")
            #print()
            #print()
        r.append(aux)
    #print(r)
    return np.array(r)

def plotting(R=1):
    z = f(x,y,R)
    #print(z)
    plt.figure(figsize=(15,10))
    ax = sb.heatmap(z,vmin=minimo,vmax=maximo,cmap="RdYlGn",annot=True,xticklabels=X, yticklabels=Y) 
    plt.title("Score over Rounds")
    plt.xlabel("Layers")
    plt.ylabel("Neuronios")
    ax.invert_yaxis()

interactive_plot = interactive(plotting,R=(0,8,1))
interactive_plot
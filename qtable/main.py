from trainqtable import *
import pandas as pd
import plotly.express as px
from tqdm import tqdm
import plotly.graph_objects as go
choice = input("1 - Treinar\n2- Jogar\n3- Plot 3D\n4 - Plot 2D\n5 - Juntar Resultados parciais\n")
if(choice == "1"):
  df = pd.DataFrame(train(rows=19))
  for i in tqdm(range(20,300)):
    df1 = train(rows=i)
    df = pd.concat([df, df1], axis=1)

  df.to_csv("results.csv")
if(choice == "2"):
  env1 = make("hungry_geese", debug=True) #set debug to True to see agent internals each step

  env1.reset()
  env1.run(["./rl-ralph.py","./submission-ralph-coward.py", "./submission-ralph-coward.py","./submission-ralph-coward.py"])

  with open('./game.html','wb') as f:   # Use some reasonable temp name
    f.write(env1.render(mode="html",width=700, height=600).encode("UTF-8"))

if(choice=="3"):
  df=pd.read_csv('results.csv', sep=',')
  #print(df.values)
  fig = go.Figure(data=[go.Surface(z=df.values[0:,1:])])
  fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                    highlightcolor="limegreen", project_z=True))
  fig.update_layout(title='Treino (X)Rows x (Y)Episódio', autosize=False,
                    width=1240, height=720,
                    margin=dict(l=65, r=50, b=65, t=90)
  )

  fig.show()
  fig.write_html("plot3d.html")
if(choice=="4"):
  i = input("Qual pasta?")
  df = pd.read_csv("./data/"+str(i)+"-0.001/results.csv")
  df = df[i]
  trace1 = px.line(
            x = df.index,
            y = df.values
            )
          
  trace1.show()
  trace1.write_html("plot2d.html")

if(choice=="5"):
  inicio = int(input("Row inicial: "))
  fim = int(input("Row Final: "))
  dfl =[]
  for i in range(inicio,fim):
    df1 = pd.read_csv("./data/"+str(i)+"-0.001/results.csv",index_col=[0])
    dfl.append(df1)
  df = pd.concat(dfl, axis=1)
  df.to_csv("results.csv")
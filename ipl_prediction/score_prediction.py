import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.losses import Huber


df=pd.read_csv('ipl_data.csv')
print(df.columns)
df=df.drop(columns=['mid','date','runs', 'wickets', 'overs','runs_last_5', 'wickets_last_5',  'striker','non-striker'])
print(df.head(5))
#'mid', 'date', 'venue', 'bat_team', 'bowl_team', 'batsman', 'bowler',
       #'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'striker',
       #'non-striker', 'total']
le=LabelEncoder()
df['venue']=le.fit_transform(df['venue'])
df['bat_team']=le.fit_transform(df['bat_team'])
df['bowl_team']=le.fit_transform(df['bowl_team'])
df['batsman']=le.fit_transform(df['batsman'])
df['bowler']=le.fit_transform(df['bowler'])

x=df.drop('total',axis=1)
y=df['total']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

scaler=MinMaxScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
y_train_scaled=scaler.fit_transform(y_train.values.reshape(-1,1)).ravel()
y_test_scaled=scaler.transform(y_test.values.reshape(-1,1)).ravel()

model=Sequential()
model.add(Input(shape=(x_train_scaled.shape[1],)))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(216,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1,activation='linear'))
model.compile(optimizer='adam',loss=Huber(delta=0.1))

model.fit(x_train_scaled,
          y_train_scaled,
          batch_size=50,
          epochs=20,
          validation_data=(x_test_scaled,y_test_scaled),
          validation_steps=40)


model_plot=pd.DataFrame(model.history.history)
model_plot.plot()

y_pred=model.predict(x_test_scaled)
y_pred=scaler.inverse_transform(y_pred)
y_test_scaled=scaler.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()
print(mean_absolute_error(y_pred,y_test_scaled))
print(mean_squared_error(y_pred,y_test_scaled))

import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore')
from IPython.display import clear_output,display

venue=widgets.Dropdown(options=df['venue'].unique().tolist(),description='select venue')
batting_team = widgets.Dropdown(options =df['bat_team'].unique().tolist(),  description='Select Batting Team:')
bowling_team = widgets.Dropdown(options=df['bowl_team'].unique().tolist(),  description='Select Bowling Team:')
batsman= widgets.Dropdown(options=df['batsman'].unique().tolist(), description='Select batsman:')
bowler = widgets.Dropdown(options=df['bowler'].unique().tolist(), description='Select Bowler:')

predict_button = widgets.Button(description="Predict Score")

def predict_score(b):
    with output:
        clear_output()
        decoded_venue = le.transform([venue.value])
        decoded_batting_team =le.transform([batting_team.value])
        decoded_bowling_team =le.transform([bowling_team.value])
        decoded_batsman =le.transform([batsman.value])
        decoded_bowler =le.transform([bowler.value])

        inputs=np.array([decoded_venue,decoded_batting_team,decoded_bowling_team,decoded_batsman,decoded_bowler])
        inputs=inputs.reshape(1,5)
        inputs=scaler.transform(inputs)

        predicted_score=model.predict(inputs)
        predicted_score=int(predicted_score[0,0])
        print(predicted_score)

predict_button.on_click(predict_score)
output = widgets.Output()
display(venue, batting_team, bowling_team,batsman, bowler, predict_button, output)


    
        































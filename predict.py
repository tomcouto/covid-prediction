import pandas as pd # data processing
from fbprophet import Prophet # Facebook's data forcasting
import os
import matplotlib.pyplot as plt # showing plots

# take state and age inputs
# age = input("Please input desired age: ")
stateIn = input("Please input desired state: ")

# load data to train prophet
train = pd.read_csv('./covid-data/us-states.csv')

# set state data
state_data = train[(train.state==stateIn)]

# Set columns for graph
state_cc = state_data[['date','deaths']]
state_cc['ds']=state_cc['date']
state_cc['y']=state_cc['deaths']
state_cc.drop(columns=['date','deaths'], inplace=True)

# Create prophet
model_cc=Prophet()
model_cc.fit(state_cc)

# Model prophet data
future = model_cc.make_future_dataframe(periods=60)

# Create future forecast
forecast=model_cc.predict(future)

# Format and show graph
fig_Confirmed = model_cc.plot(forecast,xlabel = "Date",ylabel = "Deaths")
plt.show()
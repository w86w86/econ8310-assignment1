file = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
#Import statements
import pandas as pd
import numpy as np
from prophet import Prophet

# Import the dataset of #of taxi trips each hour in New York City,
trip = pd.read_csv(file)

# Keep only the dates and the y value
trip = trip[['Timestamp','trips']]

# Format the date (Timestamp
trip.Timestamp = pd.to_datetime(trip.Timestamp)

# Prophet format using column ds and y
trip=trip.rename(columns={'Timestamp':'ds','trips':'y'})

"""###Graphic Representation of all trips (hourly)"""

import plotly.express as px
fig = px.line(trip, x='ds', y='y')
fig.show()

"""###Initialize GAM/Prophet instance and fit to data / modelFit"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# modelFit = Prophet()
# modelFit.fit(trip)

"""###Create timeline 31 days"""

model = modelFit.make_future_dataframe(periods=744, freq='h', include_history=False)

forecast = modelFit.predict(model)
pred = forecast.yhat.tolist()

"""###Dealing with the negative values"""
negative_pred = [(ind, val) \
                 for ind, val in enumerate(pred) if val<0]

"""####Replace the negative value with min of trip in an hour from our data"""

pred = [max(min(trip.y), val) for val in pred]

### 09/17  02:50

## Import statements
import pandas as pd
import numpy as np
from prophet import Prophet

## Import the dataset of #of taxi trips each hour in New York City,
file = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
trip = pd.read_csv(file)

## Keep only the dates and the y value
trip = trip[['Timestamp','trips']]
trip.Timestamp = pd.to_datetime(trip.Timestamp)

## Prophet format using column ds and y
trip=trip.rename(columns={'Timestamp':'ds','trips':'y'})

## Prophet instance and fit to data / modelFit
model = Prophet()
modelFit = model.fit(trip)

## Create timeline 31 days and forecast
future = modelFit.make_future_dataframe(periods=744, freq='h', include_history=False)
forecast = modelFit.predict(future)

pred = forecast.yhat.values

## Replace the negative value with min of trip in an hour from our data
pred = [max(min(trip.y), val) for val in pred]

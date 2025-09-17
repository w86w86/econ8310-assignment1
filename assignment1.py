

file = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
#Import statements
import pandas as pd
import numpy as np
from prophet import Prophet

# Import the dataset of #of taxi trips each hour in New York City,
trip = pd.read_csv(file)

# Keep only the dates and the y value
trip = trip[['Timestamp','trips']]
print('->trip.Timestamp (before) dtype: ', trip.Timestamp.dtype)

# Format the date (Timestamp
trip.Timestamp = pd.to_datetime(trip.Timestamp)

# Prophet format using column ds and y
trip=trip.rename(columns={'Timestamp':'ds','trips':'y'})

#Create timeline 31 days
future = modelFit.make_future_dataframe(periods=744, freq='h', include_history=False)
print('future.shape: ', future.shape)

forecast = modelFit.predict(future)
#forecast
pred = forecast.yhat.tolist()

#Dealing with the negative values
negative_pred = [(ind, val) \
                 for ind, val in enumerate(pred) if val<0]
print ('Nb of negative values:', len(negative_pred))

#Replace the negative value with min of trip in an hour from our data
pred = [max(min(trip.y), val) for val in pred]

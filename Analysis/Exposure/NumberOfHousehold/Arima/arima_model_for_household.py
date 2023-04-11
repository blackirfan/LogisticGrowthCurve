import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Input the data as a list of tuples
# data = [(2001, 36308), (2011, 44947), (2022, 58298)]
# HouseHold      
# data = [(2001, 6944), (2011, 8786), (2022, 47588)] # Amin Bazar
# data = [(2001, 5046), (2011, 7742), (2022, 40136)] # Banagram
# data = [(2001, 7422), (2011, 9366), (2022, 58298)] # Bhakurta
# data = [(2001, 4947), (2011, 9829), (2022, 69330)] # Biralia
# data = [(2001, 4277), (2011, 6107), (2022, 35596)] # Kaundia
# data = [(2001, 3555), (2011, 11575), (2022, 65118)] # Savar
# data = [(2001, 28737), (2011, 73465), (2022, 384105)] # Savar Paurashava
data = [(1981, 5103),(1991, 4634),(2001, 8506), (2011, 25867), (2022, 268229)] # Tetuljhora
# Convert the data to a pandas DataFrame
df = pd.DataFrame(data, columns=['Year', 'Population'])

# Set the index of the DataFrame to the 'Year' column
df = df.set_index('Year')

# Fit an ARIMA model to the data
model = ARIMA(df, order=(1, 1, 1))
model_fit = model.fit()

# Generate population forecasts for the years 2030, 2050, 2070, 2090, and 2100
index = pd.date_range(start='2030', end='2100', freq='AS')
forecast = model_fit.forecast(len(index))

# Convert the forecasted population values to a numpy array
forecast = np.array(forecast)

# Print the predicted population values
for i in range(len(index)):
    year = index[i].year
    population = int(forecast[i])
    if(year == 2030):
        print(f"Population prediction for {year} year: {population}")
    if(year == 2050):
        print(f"Population prediction for {year} year: {population}")
    if(year == 2070):       
        print(f"Population prediction for {year} year: {population}")
    if(year == 2090):
        print(f"Population prediction for {year} year: {population}")
    if(year == 2100):
        print(f"Population prediction for {year} year: {population}")



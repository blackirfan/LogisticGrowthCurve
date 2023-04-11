import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Input the data as a list of tuples
# Population
# data = [(1981, 17064),(1991, 23528),(2001, 29991), (2011, 37500), (2022, 47589)] # Amin Bazar
# data = [(1981, 16697),(1991, 20840),(2001, 24742), (2011, 33627), (2022, 40136)] # Banagram
# data = [(1981, 4427),(1991, 34452),(2001, 36308), (2011, 44947), (2022, 58298)] # Bhakurta

# data = [(1981, 15521),(1991, 18478),(2001, 23760), (2011, 41188), (2022, 69330)] # Biralia

# data = [(1981, 11975),(1991, 14414),(2001, 20065), (2011, 27796), (2022, 35596)] # Kaundia

# data = [(1981, 42236),(1991, 29544),(2001, 16851), (2011, 45887), (2022, 65115)] # Savar


data = [(1981, 22273),(1991, 106586),(2001, 127540), (2011, 296851), (2022, 384103)] # Savar Paurashava

# data = [(1981, 20413),(1991, 28192),(2001, 41978), (2011, 106929), (2022, 268229)] # Tetuljhora
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
answer = ""
# Print the predicted population values
for i in range(len(index)):
    year = index[i].year
    population = int(forecast[i])
    if(year == 2030):
        print(f"Population prediction for {year} year: {population}")
        answer = answer +" "+ str(population)
    if(year == 2050):
        print(f"Population prediction for {year} year: {population}")
        answer = answer +" "+ str(population)
    if(year == 2070):       
        print(f"Population prediction for {year} year: {population}")
        answer = answer +" "+ str(population)
    if(year == 2090):
        print(f"Population prediction for {year} year: {population}")
        answer = answer +" "+ str(population)
    if(year == 2100):
        print(f"Population prediction for {year} year: {population}")
        answer = answer +" "+ str(population)
print(answer)


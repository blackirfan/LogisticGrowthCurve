# import numpy as np
# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt
# import pandas as pd
# import math

# # Define the logistic function
# def logistic_func(x, K, r, x0):
#     prediction_result = K / (1 + np.exp(-r * (x - x0)))
#     return prediction_result

# # Define the initial parameters
# xdata = np.array([2001,2011,2022])

# ydata = np.array([7422,	9366,	12148])

# # Calculate the mean and standard deviation of the population data
# pop_mean = np.mean(ydata)
# pop_std = np.std(ydata)

# # Estimate K and r from the mean and standard deviation
# #Carring Capacity
# K = (2*ydata[0]*ydata[1]*ydata[2] - ydata[1]*ydata[1]*(ydata[0] + ydata[2]) )/(ydata[0]*ydata[2] - ydata[1]*ydata[1])
# print("Carring Capacity : ",K)
# P_0 = ydata[0]
# P_1 = ydata[1]
# P_s = K
# t_1 = 10
# #Rate
# r = (2.3 * math.log10((P_0 * (P_s - P_1))/(P_1 * (P_s - P_0))))/t_1
# # r = 4 * pop_std / pop_mean
# # Find the year with the maximum population growth rate
# # x0 = [np.argmax(np.diff(ydata))]
# x0 = 2022
# # Set the initial parameter values
# p0 = [K, r, x0]


# # Fit the logistic function to the data
# popt, pcov = curve_fit(logistic_func, xdata, ydata, p0)

# # Use the fitted model to predict the population for future years
# years = np.array([2030, 2040, 2050, 2060,2070,2080,2090,2100])
# population_pred = logistic_func(years, *popt)

# # Create a DataFrame with the predicted population data
# data = {'Year': years, 'Household': population_pred}
# df = pd.DataFrame(data)

# # Save the DataFrame as a CSV file
# df.to_csv('number_of_household_Bhakurta.csv', index=False)

# # Plot the original data and the predicted values
# plt.plot(xdata, ydata, 'o', label='Original data')
# plt.plot(years, population_pred, 'r-', label='Predicted values')
# plt.xlabel('Year')
# plt.ylabel('Number of Household')
# plt.title('Predicted Number of Household in Bhakurta')
# plt.legend()
# plt.show()
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define the data
xdata = np.array([2001, 2011, 2022])
ydata = np.array([7422, 9366, 12148])

# Convert the data to a pandas DataFrame with a datetime index
dates = pd.date_range(start='2001', end='2022', freq='y')
data = pd.DataFrame({'population': ydata}, index=dates)

# Split the data into training and testing sets
train = data.iloc[:-1]
test = data.iloc[-1:]

# Identify the order of differencing (d), autoregressive term (p), and moving average term (q)
d = 1
p = 1
q = 1

# Fit the ARIMA model to the training data
model = ARIMA(train, order=(p, d, q))
results = model.fit()

# Generate a forecast for the test data
forecast = results.forecast(steps=len(test))[0]

# Evaluate the accuracy of the model
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = np.mean(np.abs((test - forecast) / test)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
# Assuming you have already fitted an ARIMA model and stored the results in the variable `results`

# Generate a forecast for the year 2030
forecast = results.forecast(steps=1)

print(f"Forecast for 2030: {forecast[0]:.0f}")
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import math
np.seterr(over='ignore')
# Define the logistic function
# def logistic_func(x, K, r, x0):
#     prediction_result = K / (1 + np.exp(-r * (x - x0)))
#     return prediction_result

# Define the initial parameters
xdata = np.array([2001, 2011, 2022])

ydata = np.array([127540,296851,384103])

# Calculate the mean and standard deviation of the population data
pop_mean = np.mean(ydata)
pop_std = np.std(ydata)

# Estimate K and r from the mean and standard deviation
#Carring Capacity
print(ydata[0])
print(ydata[1])
print(ydata[2])
n_1 = (2*ydata[0]*ydata[1]*ydata[2])
n_2 = ydata[1]*ydata[1]*(ydata[0] + ydata[2])
d_1 = (ydata[0]*ydata[2])
d_2 = (ydata[1]*ydata[1])
print(n_1)
print(n_2)
print(d_1)
print(d_2)
numerator_1_2 = n_1 - n_2
denominator_1_2 = d_1 - d_2
print(numerator_1_2)
print(denominator_1_2)
numerator = (2*ydata[0]*ydata[1]*ydata[2]) - (ydata[1]*ydata[1]*(ydata[0] + ydata[2]))
denominator = (ydata[0]*ydata[2]) - (ydata[1]*ydata[1])
p0p1p2 = 2*ydata[0]*ydata[1]*ydata[2]
print("numerator : ",numerator)
print("denominator : ",denominator)
carring_capacity = numerator / denominator
print("carring_capacity : " ,carring_capacity)
print(p0p1p2)
K = ((2*ydata[0]*ydata[1]*ydata[2]) - (ydata[1]*ydata[1]*(ydata[0] + ydata[2])) )/((ydata[0]*ydata[2]) - (ydata[1]*ydata[1]))
print("Carring Capacity : ",K)
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
# data = {'Year': years, 'Population': population_pred}
# df = pd.DataFrame(data)

# # Save the DataFrame as a CSV file
# df.to_csv('population_density_SavarPaurashava.csv', index=False)

# # Plot the original data and the predicted values
# plt.plot(xdata, ydata, 'o', label='Original data')
# plt.plot(years, population_pred, 'r-', label='Predicted values')
# plt.xlabel('Year')
# plt.ylabel('Population Density')
# plt.title('Predicted Population Density of Savar Paurashava')
# plt.legend()
# plt.show()
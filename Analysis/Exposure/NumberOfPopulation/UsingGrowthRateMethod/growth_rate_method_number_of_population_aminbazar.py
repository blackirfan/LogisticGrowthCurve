import csv



# Define the starting population
P0 = 37500	

P1 = 47589

# Define the growth rate (as a decimal)
r = ((P1/P0)**(1/10) - 1)

# Define a list of years for which we want to estimate population
years = [2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]

# Create a list to store the estimated population values
population_estimates = []

# Loop through each year and calculate the estimated population
for year in years:
    t = year - 2011
    P = P0 * (1 + r)**t
    population_estimates.append(P)

# Save the population estimates to a CSV file
with open("population_estimates_aminbazar.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Year", "Population"])
    for i, year in enumerate(years):
        writer.writerow([year, population_estimates[i]])
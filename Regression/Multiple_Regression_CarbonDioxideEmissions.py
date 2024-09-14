import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


df = pd.read_csv("../Datasets/FuelConsumptionCo2.csv")

# shows frist lines
head = df.head()

# shows some basic statistics
statistics = df.describe()

# for visualization use matplotlib
#plt.scatter(df.FUELCONSUMPTION_COMB, df.CO2EMISSIONS,  color='blue')
#plt.xlabel("FUELCONSUMPTION_COMB")
#plt.ylabel("Emission")
#plt.show()

# Split Data in Training and Test Set
msk = np.random.rand(len(df)) < 0.8 # Is a list of True or False, to divide the rows
train = df[msk]
test = df[~msk]

# Model Data with Linear Regression -> Calculates the Linear Function mx+b
# Therefore specify dependend & independend data

# regr.fit() Trains the Model

from sklearn import linear_model
regr = linear_model.LinearRegression()
# training data
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print('Coefficients: ', regr.coef_) # m in mx+b
print('Intercept: ', regr.intercept_) # b in mx+b

# Plot Output
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# Take Test Set
# Predict value and Evaluate Results

test_x = np.asanyarray(test[['ENGINESIZE']]) #
test_y = np.asanyarray(test[['CO2EMISSIONS']]) # Actual Values
test_y_ = regr.predict(test_x) # Predicted Values

# Compare Actual and Predicted Values
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))


print()


import numpy as np
import math
import csv


import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

rows = []
data = list(open('newtest.csv', 'r'))
for i in data:
	rows.append(i.split("\n")[:-1])
actual_rows = []
temp_rows = []
for k in rows:
	for j in k:
		actual_rows.append(j.split("\t"))

actual_rows.pop(0)
actual_rows = np.array(actual_rows)
size = 253 ##to get the depth of tree for any set of data.
t_size= 52
actual_rows= np.float32(actual_rows)
values = actual_rows.astype('float32')
train = values[0:200, :]
test = values[200:size, :]
train_X = train[:, 2:]
train_y = train[:, [0,1]]
test_X = test[:, 2:]
actual = test[:, [0,1]]
depth = size/15  ## Equation for depth of tree
regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=depth, random_state = 0))
regr_multirf.fit(train_X, train_y)

regr_rf = RandomForestRegressor(max_depth=depth, random_state=2)
regr_rf.fit(train_X, train_y)

y_rf = regr_rf.predict(test_X)
p = regr_multirf.predict(test_X)

print("MultiOutputRegressor predictions:\n", p)
print("Mean Squared Error for Multi-Output Regressor : ")
print(sqrt(mean_squared_error(actual, p)))

errors = abs(p - actual)
mape = 100 * (errors / actual)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Mean Accuracy:', round(accuracy, 2), '%.')

print("---------------------------------------------------------------------------------------------------------")

print("Random Forest Predictions: \n", y_rf)
print("Mean Squared Error for Random Forest Regressor: ")
print(sqrt(mean_squared_error(actual, y_rf)))
#print(type(train_X))

errors = abs(y_rf - actual)
mape = 100 * (errors / actual)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Mean Accuracy:', round(accuracy, 2), '%.')

print("---------------------------------------------------------------------------------------------------------")
print("Difference For RandomForestRegressor")
for i in range(t_size):
	difference = list([(actual[i][0]-y_rf[i][0]),(actual[i][1]-y_rf[i][1])])
	print(difference)

print("---------------------------------------------------------------------------------------------------------")
print("Difference For MultiOutputRegressor")
for i in range(t_size):
	difference = list([(actual[i][0]-p[i][0]),(actual[i][1]-p[i][1])])
	print(difference)
	
for i in range(t_size):
	plt.plot(y_rf[i][0],y_rf[i][1], 'ro', label = "RandomForestRegressor")	

for i in range(t_size):
	plt.plot(actual[i][0], actual[i][1], 'bs', label = "actual data")
		
plt.xlabel("Latitute")
plt.ylabel("Longitude")
axes = plt.gca()
axes.set_xlim([12.275,12.375])
axes.set_ylim([76.5,76.8])
plt.show()

for i in range(t_size):
	plt.plot(p[i][0], p[i][1], 'g^', label="MultiOutputRegressor")

for i in range(t_size):
	plt.plot(actual[i][0],actual[i][1], 'bs', label="actualData")
		
plt.xlabel("Latitute")
plt.ylabel("Longitude")
axes = plt.gca()
axes.set_xlim([12.275,12.375])
axes.set_ylim([76.5,76.8])
plt.show()
for i in range(t_size):
	plt.plot(p[i][0], p[i][1], 'g^', label="MultiOutputRegressor")
		
for i in range(t_size):
	plt.plot(y_rf[i][0],y_rf[i][1], 'ro', label = "RandomForestRegressor")
plt.xlabel("Latitute")
plt.ylabel("Longitude")
axes = plt.gca()
axes.set_xlim([12.275,12.375])
axes.set_ylim([76.5,76.8])
plt.show()

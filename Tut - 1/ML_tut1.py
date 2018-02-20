import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# data-frame variable "fruits".
fruits = pd.read_table('fruit_data_with_colors.txt')

#making a lookup dict type to render label as fruit name.
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
print lookup_fruit_name

X = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']

# splitting the dataset in 75 % train -25 % test ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
print fruits.head()

# initialising a classifier
knn = KNeighborsClassifier(n_neighbors = 5)
# training the classifier
knn.fit(X_train, y_train)

# printing the test efficiency.
print "Efficency of the test: ", knn.score(X_test, y_test)

# predicting a label, by [[mass, width, height]].
fruit_prediction = knn.predict([[20, 4.5, 9]])
# look for the fruit name in lookup dictionary.
print lookup_fruit_name[fruit_prediction[0]]


# plotting acuracy curve
k_range = range(1, 20)
efficiency = []
kval = []

for i in k_range:
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)

    eff = knn.score(X_test, y_test)

    kval.append(i)
    efficiency.append(eff)

plt.scatter(kval, efficiency)
plt.show()

# NOTE: High k values results in significant reduction of noise but also decreases the accuracy.

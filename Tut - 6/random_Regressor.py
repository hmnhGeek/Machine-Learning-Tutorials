from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('radm.csv')

X = []
for i in range(len(df.Feature.values)):
    X.append([df.Target1.values[i], df.Target2.values[i]])

y = df.Feature.values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train, y_train)

target1 = float(raw_input("Enter Target1: "))
target2 = float(raw_input('Enter Target2: '))

print knn.predict(np.array([target1, target2]).reshape(1, -1))[0]

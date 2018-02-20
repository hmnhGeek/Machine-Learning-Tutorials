from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# open the dataset using pandas
df = pd.read_csv('buildings_data.csv')

compactness = [[i] for i in df.X1.values]
surface_ar = [[i] for i in df.X2.values]
wall_ar = [[i] for i in df.X3.values]
roof_ar = [[i] for i in df.X4.values]
height = [[i] for i in df.X5.values]
orientation = [[i] for i in df.X6.values]
glazing_ar = [[i] for i in df.X7.values]
glazing_dist = [[i] for i in df.X8.values]

X = []

for i in range(len(compactness)):
    X.append([float(compactness[i][0]), float(surface_ar[i][0]), float(wall_ar[i][0]), float(roof_ar[i][0]), float(height[i][0]), \
              float(orientation[i][0]), float(glazing_ar[i][0]), float(glazing_dist[i][0])])

y = df.label.values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def accuracy_curve():
    neighbors = range(1, 20)
    knn_axis, clf_axis = [], []
    for neighbor in neighbors:
        knn = KNeighborsRegressor(n_neighbors=neighbor)
        knn.fit(X_train, y_train)
        clf = KNeighborsClassifier(n_neighbors=neighbor)
        clf.fit(X_train, y_train)
        knn_axis.append(knn.score(X_test, y_test))
        clf_axis.append(clf.score(X_test, y_test))

    plt.plot(neighbors, knn_axis, '-o', label = "Regression accuracy")
    plt.plot(neighbors, clf_axis, '-o', label = "Classifier accuracy")
    plt.legend()
    plt.xlabel('k - value')
    plt.ylabel('Accuracy on 0 to 1 scale')
    plt.grid()
    plt.title("Buildings heating load and cooling load prediction")
    plt.show()
        
accuracy_curve()

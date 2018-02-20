from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loading the .csv dataset
df = pd.read_csv('buildings_data.csv')

# store features
rel_compactness = df[['X1']].values
surface_ar = df[['X2']].values
wall_ar = df[['X3']].values
roof_ar = df[['X4']].values
overallHt = df[['X5']].values
orientation = df[['X6']].values
glazing_ar = df[['X7']].values
glazing_dist = df[['X8']].values

X = []
for i in range(len(rel_compactness)):
    X.append([float(rel_compactness[i][0]), float(surface_ar[i][0]), float(wall_ar[i][0]), float(roof_ar[i][0]), float(overallHt[i][0]), \
    float(orientation[i][0]), float(glazing_ar[i][0]), float(glazing_dist[i][0])])

y = df['label']

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# initialise and train the classififer

neighbors = range(1, 20)
eff = []

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)
print "Test is efficient with efficiency", knn.score(X_test, y_test)

print
print "Enter Building details..."
rel_comp = input("Relative compactness: ")
SA = input("Surface area: ")
WA = input("Wall area: ")
RA = input("Roof area: ")
OH = input("Overall height: ")
OR = input("Orientation: ")
GA = input("Glazing Area: ")
GD = input("Glazing Distribution: ")

l= knn.predict(np.asarray([rel_comp, SA, WA, RA, OH, OR, GA, GD]).reshape(1, -1))

y1 = dict(zip(df.label.unique(), df.Y1.unique()))
y2 = dict(zip(df.label.unique(), df.Y2.unique()))
print "Heating Load:", y1[l[0]], "Cooling Load:", y2[l[0]]

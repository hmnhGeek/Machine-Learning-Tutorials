from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("flavors_of_cacao.csv")
cocpercent = df[[u'Cocoa\nPercent']].values
ratings = df[[u'Rating']].values

companies = df[u'Company']
X = []

for i in range(len(cocpercent)):
    X.append([float(cocpercent[i][0].rstrip('%')), float(ratings[i][0])])

X_train, X_test, y_train, y_test = train_test_split(X, companies, random_state = 0)
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, y_train)

print "Classifier trained!!"
print

cocoa_percent = float(raw_input("Enter Cocoa Percent (%): "))
rati = float(raw_input("Enter rating: "))

print knn.predict([[cocoa_percent, rati]])[0]

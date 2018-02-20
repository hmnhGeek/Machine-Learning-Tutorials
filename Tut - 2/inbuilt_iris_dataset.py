from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

print
print "Welcome to Iris..."
print "=================="
print

print
print "Analysing efficiency..."
print knn.score(X_test, y_test)
print

sl = float(input("Sepal Length (cm): "))
sw = float(input("Sepal Width (cm): "))
pl = float(input("Petal length (cm): "))
pw = float(input("Petal Width (cm): "))

prediction_param = knn.predict([[sl, sw, pl, pw]])

if prediction_param[0] == 0:
    print 'sesota'
elif prediction_param[0] == 1:
    print 'versicolor'

else:

    print 'virginica'
    

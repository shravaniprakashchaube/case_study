# consider below application which loads iris dataset and display all records and labels of that data set
#classifier = Decision Tree
#dataset= iris datatset
#Features = sepal length ,petal length, sepal width,petal width
#labels = versicolor,setosa,virginica

from sklearn.datasets import load_iris
iris = load_iris()
print("features names of iris data set")
print(iris.feature_names)
print("Target names of iris data set")
print(iris.target_names)
print("first 10 elements from iris data set")
for i in range(len(iris.target)):
    print("ID:%d,Label%s,Feature:%s"%(i,iris.data[i],iris.target[i]))

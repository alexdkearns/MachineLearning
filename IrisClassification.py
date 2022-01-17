import mglearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

#  Imports relevant packages. pip installed conda then pip installed mglearn could not load package from pycharm

iris_dataset = load_iris()  # Loads data set and sets it as a variable similar to a dict
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys())) # Prints data keys

print(iris_dataset['DESCR'][:193] + "\n...")  # Prints a short description of the dataset

print("Target names: {}".format(iris_dataset['target_names']))  # Target names are strings that contain iris species types

print("Feature names: \n{}".format(iris_dataset['feature_names']))  # Feature names give a desc string of each feature

print("Type of data: {}".format(type(iris_dataset['data'])))  # Datatype = np.array; rows=flowers cols=four measurements

print("Shape of data: {}".format(iris_dataset['data'].shape))  # 150 different flowers and four measurement cats 150,4

print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))  # Prints first 5 rows

print("Type of target: {}".format(type(iris_dataset['target'])))  # Target is 1D array with one entry per flower

print("Shape of target: {}".format(iris_dataset['target'].shape))  # The species are encoded as integers from 0 to 2

print("Shape of target: {}".format(iris_dataset['target']))  # 0;Setosa, 1;Versicolr, 2;Virginica

from sklearn.model_selection import train_test_split  # Imports Functions

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)  # Calls function and assigns outputs

print("X_train shape: {}".format(X_train.shape))  # Contains 75% of the rows in the dataset
print("y_train shape: {}".format(y_train.shape))  # Contains 75% of the rows in the dataset
print("X_train shape: {}".format(X_test.shape))  # Contains remaining 25% of rows in the dataset
print("y_train shape: {}".format(y_test.shape))  # Contains 75% of the rows in the dataset

# Inspecting data

# Create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, colour by y_train
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                     hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

plt.show()


    ## FIRST MODEL (k-Nearest Neighbours) ##

from sklearn.neighbors import KNeighborsClassifier  # Imports ML Model
knn = KNeighborsClassifier(n_neighbors=1)  # The knn object holds the algo that will be built using the training data

knn.fit(X_train, y_train)  # Fits model using training data

X_new = np.array([[5, 2.9, 1, 0.2]])  # Makes new np.array with data e.g. 5cm sepal length 2.9cm sepal width etc
print("X_new.shape: {}".format(X_new.shape))
prediction = knn.predict(X_new)  # Call the predict method of the knn object on the new data
print("Prediction: {}".format(prediction))
print("Prediction target name: {}".format(
    iris_dataset['target_names'][prediction]))  # The prediction is given as 0 index = Setosa species

    ## EVALUATING THE MODEL ##

y_pred = knn.predict(X_test)  # calls predict on the test set of data
print("Test set predictions:\n {}".format(y_pred))

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))   # compares the prediction and test data acc.

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))  # can also use score method of knn object to get acc.

# Summary of the code needed for the entire training and evaluation procedure
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


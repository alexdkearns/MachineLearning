# Generate dataset
# Imports relevant packages
import matplotlib.pyplot as plt
import numpy as np
import mglearn
from mglearn import datasets
from mglearn import plot_helpers

X, y = mglearn.datasets.make_forge()
# plot dataset
plot_helpers.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)  # Creates and Names legend to describe datapoints
plt.xlabel("First feature")  # Labels x-axis
plt.ylabel("Second feature")  # Labels y-axis
print("X.shape: {}".format(X.shape))  # shape of data 26 rows and two features(columns)
plt.show()  # Makes plot pop up
X, y = mglearn.datasets.make_wave(n_samples=40)  # Imports dataset from mglearn package
plt.plot(X, y, 'o')  # plots X and y data points and uses the 'o' plot points
plt.ylim(-3, 3)  # Places limits on y dataset
plt.xlabel("Feature")  # Labels x axis label
plt.ylabel("Target")  # Labels y axis label
plt.show()
from sklearn.datasets import load_breast_cancer  # Imports package from sklearn

cancer = load_breast_cancer()  # assigns dataset to variable for future use
print("cancer.keys(): \n{}".format(cancer.keys()))  # Prints keys of the dataset
print("Shape of cancer data: {}".format(cancer.data.shape))  # prints rows and features
print("Sample counts per class:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

print("Feature names:\n{}".format(cancer.feature_names))  # prints feature names in dataset
# Imports boston housing dataset
import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
print("Data shape: {}".format(data.data.shape))

# Imports extended boston dataset
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))  # Prints shape of dat 504 rows and 104 features 13 original + 91 combos of the 13
from mglearn import plots

mglearn.plots.plot_knn_classification(n_neighbors=1)  # k-NN algo will consider nearest neighbor only
plt.show()

mglearn.plots.plot_knn_classification(n_neighbors=3)  # k-NN algo will consider three nearest neighbors
plt.show()

from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train, y_train)

print("Test set predictions: {}".format(clf.predict(X_test)))

print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    # the fit method returns the object self, so we can instantiate
    # and fit in one line
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    plot_helpers.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()

# K-Neighbours regression

mglearn.plots.plot_knn_regression(n_neighbors=1)  # plots single nearest neighbour using wave dataset
plt.show()  # prediction using single neighbour is just the target value of the nearest neighbour
mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()  # same plot when using multiple nearest neighbours the prediction is the average/mean of neighbours

from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)

# split the wave dataset into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# instantiate the model and set the number of neighbours to consider to 3
reg = KNeighborsRegressor(n_neighbors=3)
# fit the model using the training data and training targets
reg.fit(X_train, y_train)  # Fits model
print("Test set predictions:\n{}".format(reg.predict(X_test)))  # runs prediction

print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))  # gives accuracy

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# create 1,000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.plot_helpers.cm2(0), markersize=8)
    ax.plot(X_test, y_test, '^', c=mglearn.plot_helpers.cm2(1), markersize=8)

    ax.set_title(
        "{} neighbour(s)\n train score: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_xlabel("Target")
axes[0].legend(["Model predictions", "Training data/target",
                "Test data/target"], loc="best")
plt.show()  # This shows the comparing predictions made by nearest neighbors regression for diff values of n_neighbors


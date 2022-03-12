# Linear models make a prediction using a linear function of the input features.
# General formula: y = w[0] * w[1] * x[1] + ... + w[p] * x[p] + b
# x[0] & x[p] denote features No. of features = p of single datapoint, w, b are params of the model that are learned
# y = prediction the model makes. For a dataset with a single feature: y = w[0] * x[0] + b

import matplotlib.pyplot as plt
import numpy as np
import mglearn
from mglearn import plots
from mglearn import datasets
from mglearn import plot_helpers
from sklearn.model_selection import train_test_split

mglearn.plots.plot_linear_regression_wave()
plt.show()
# Can see from plot w[0] aka the slope is around 0.4. The intercept is where the prediction line should cross


# Linear regression (aka ordinary least squares)
from sklearn.linear_model import LinearRegression

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test, = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

print("Training set score {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2}".format(lr.score(X_test, y_test)))
# Scores indicate underfitting

X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
print("Training set score {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2}".format(lr.score(X_test, y_test)))
# Ridge Regression

from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print("Training set score {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2}".format(ridge.score(X_test, y_test)))

ridge10 = Ridge(alpha=10).fit(X_train, y_train) # alpha = importance of simplicity vs training set performance
print("Training set score {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2}".format(ridge10.score(X_test, y_test)))

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train) # alpha = importance of simplicity vs training set performance
print("Training set score {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.2}".format(ridge01.score(X_test, y_test)))

plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
plt.show()

mglearn.plots.plot_ridge_n_samples()
plt.show()

from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

# we increase the default setting of "max_iter",
# otherwise the model would warn us that we should increase max_iter.
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ !=0)))
# lower alpha allows the fit of more complex models which worked better on both data-sets,
# Performance slightly better than ridge while only using 33/105 features

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ !=0)))

# Plot the coefficients of the different models
plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")

plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.show()

# For a=1 most coefficients are zero
# a=0.01 causes most features to be zero
# a=0.00001 model is not regularised
# Ridge a=0.1 has similar predictive performance as the lasso model with alpha=0.01


# LINEAR MODELS FOR CLASSIFICATION
# Binary Classification: y=w[0] * x[0] * w[1]+...+w[p] * x[p] + b > 0

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
                                    ax=ax, alpha=.7)
    mglearn.plot_helpers.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()
plt.show()
# Decision boundaries of a linearSVM and logistic regression on the forge dataset with the default params
mglearn.plots.plot_linear_svc_regularization()
plt.show()
# Decision boundaries of a linearSVM on the forge dataset for different values of C


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression(max_iter=10000).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))
# C=1 provided good performance but since the two scores are similar this is likely overfitting

logreg100 = LogisticRegression(C=100, max_iter=10000).fit(X_train, y_train)
print("Training set score {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))
# C=100 results in higher training accuracy confirming assumption that more complex model = better performance

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))
# As expected, when moving to C=0.01 w/ an already underfitted model both training and test set acc decrease

plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.01")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.show()


for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l2 = LogisticRegression(C=C, penalty="l2", max_iter=10000).fit(X_train, y_train)
    print("Training accuracy of l2 logreg with C={:.3f}: {:.2f}".format(
        C, lr_l2.score(X_train, y_train)))
    print("Test accuracy of l2 logreg with C={:.3f}: {:.2f}".format(
        C, lr_l2.score(X_test, y_test)))
    plt.plot(lr_l2.coef_.T, marker, label="C={:.3f}".format(C))

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")

plt.ylim(-5, 5)
plt.legend(loc=3)
plt.show()



#  LINEAR MODELS FOR MULTICLASS CLASSIFICATION
# w[0] * x[0] + w[1] * x[1] + ... + w[p] * X[p] + b

from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)
mglearn.plot_helpers.discrete_scatter(X[:,0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.xlabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])
plt.show()


# Train linearSVC classifier on the dataset

linear_svm = LinearSVC().fit(X, y)
print("Coefficient shape: ", linear_svm.coef_.shape)
print("Intercept shape: ", linear_svm.intercept_.shape)

mglearn.plot_helpers.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
           'Line class 2'], loc=(1.01, 0.3))
plt.show()
# shows Decision boundaries learned by the three one vs rest classifiers

mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.plot_helpers.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
           'Line class 2'], loc=(1.01, 0.3))
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

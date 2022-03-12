from mglearn import plot_helpers
import mglearn
from mglearn import make_blobs
from sklearn.datasets import make_blobs
from mglearn import plots
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D, axes3d
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Kernelised Support Vector Machines
# Linear Models and nonlinear features
X, y = make_blobs(centers=4, random_state=8)
y = y % 2

mglearn.plot_helpers.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
# Two-class classification dataset in which classes are not linearly separable

linear_svm = LinearSVC().fit(X, y)

mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.plot_helpers.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
# Decision boundary found by a linear SVM
# Now expand set of input features, say by also adding feature1 ** 2, the square of the second feature as a new feature
# as a two dimensional point, (feature 0, feature 1), we now represent it as a three-dimensional point
# (feature0, feature1, feature1 ** 2).

# add the squared first feature
X_new = np.hstack([X, X[:, 1:] ** 2])


figure = plt.figure()
# Visualise in 3D
ax = Axes3D(figure, elev=-152, azim=26)
# plot first all the points with y==0, then all with y==1
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
           cmap=mglearn.plots.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
           cmap=mglearn.plots.cm2, s=60)
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")
plt.show()

# Expansion of the dataset shown before but adding a third feature derived from feature1

linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

# show linear decision boundary
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)

XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
           cmap=mglearn.plots.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
           cmap=mglearn.plots.cm2, s=60)
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")
plt.show()
# Decision boundary found by a linear SVM on the expanded three-dimensional dataset

ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
             cmap=mglearn.plots.cm2, alpha=0.5)
mglearn.plot_helpers.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

# The Kernel Trick
# Adding nonlinear functions to the representation of data can make linear models more powerful
# This trick lets you learn a classifier in a higher-dimensional space without actually computing the new representation

# Polynomial kernel computes all possible polynomials up to a certain degree of the original features( f1 ** 2 * f2 **5)
# Radial basis function (RBF) kernel aka gaussian kernel


# Understanding SVMs - SVM learns how important each of the training data points is to the decision boundary
# defines decision boundary the ones that lie on the border between the classes are support vectors hence SVM
# To make a pred for a new point, the distance to each of the support vectors is measured
# Distance = krbf(xp, x2) = exp(gamma|x1-x2||**2)
# x1, x2 are data points. ||x1-x2|| denotes Euclidean distance, and gamma is a param that controls the width of kernel

from sklearn.svm import SVC
from mglearn import tools
X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.plot_helpers.discrete_scatter(X[:, 0], X[:, 1], y)
# plot support vectors
sv = svm.support_vectors_
# class labels of support vectors are given by the sign of the dual coefficients
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.plot_helpers.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
# Decision boundary and support vectors found by an SVM with RBF kernel
# SVM yielded a very smooth non-linear boundary. We adjust two params C and gamma

# Tuning SVM parameters
# gamma controls width of Gaussian kernel aka the scale of what it means to be close to one another
# The C param is for regularisation similar to that used in the linear models it limits the importance of each point

fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for ax, C in zip(axes, [-1, 0 ,3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)

axes[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"],
                  ncol=4, loc=(.9, 1.2))
plt.show()
"""
from L to R we increase gamma 0.1 - 10. A small gamma means a larger Gaussian Kernel this means many points are
considered close by. This is reflected by very smooth decision boundaries on the left and boundaries that focus more
on single points further to the right.
A low value of gamma means that the decision boundary will vary slowly, which yields a model of low complexity
the opposite is also true.
from top to bottom, we increase the C param from 0.1 to 1000. As with the linear models a small C means a restricted
model where each data point has limited influence. you can see top left hand corner the boundary is very linear
with the misclassified points not having much of an impact and as you go down you see the points have more of an 
influence on the line.
"""

# RBF kernel SVM applied to the Breast Cancer dataset. By default, C=1 and gamma=1/n_features
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

svc = SVC()
svc.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))

plt.plot(X_train.min(axis=0), 'o', label="min")
plt.plot(X_train.max(axis=0), '^', label="max")
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")
plt.show()
# Feature ranges for the Breast Cancer Dataset (y axis with log scale)

# Preprocessing data for SVMs
# Rescale each feature so that they are all approximately on the same scale for kernel SVMs scale data such that all
# features are between 0 and 1

# compute the minimum value per feature on the training set
min_on_training = X_train.min(axis=0)
# compute the range of each feature (max-min) on the training set
range_on_training = (X_train - min_on_training).max(axis=0)

# subtract the min, and divide by range
# afterward, min=0 and max=1 for each feature
X_train_scaled = (X_train - min_on_training) / range_on_training
print("Minimum for each feature\n{}".format(X_train_scaled.min(axis=0)))
print("Maximum for each feature\n{}".format(X_train_scaled.max(axis=0)))

# use THE SAME transformation on the test set
# using min and range of the training set
X_test_scaled = (X_test - min_on_training) / range_on_training
svc = SVC()
svc.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
    svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))
# Increasing C allows significant improvement in the model

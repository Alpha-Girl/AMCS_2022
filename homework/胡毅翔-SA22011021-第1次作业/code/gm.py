from sklearn import datasets
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
# load data
X, y = datasets.load_iris(return_X_y=True)

# Gaussian Mixture Model
# set covariance_type as spherical, diagonal, tied or full
gm = GaussianMixture(n_components=3, covariance_type='full',
                     random_state=0).fit(X, y)

# print mean, variance and weight
print("mean : ", gm.means_)
print("variance : ", gm.covariances_)
print("weight : ", gm.weights_)

# predict
labels = gm.predict(X)

# plot
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
for i in range(len(y)):
    if y[i] == 0:
        ax.scatter(X[i][0], X[i][1], X[i][2], c='r')
    elif y[i] == 1:
        ax.scatter(X[i][0], X[i][1], X[i][2], c='blue')
    elif y[i] == 2:
        ax.scatter(X[i][0], X[i][1], X[i][2], c='g')
ax.set_title("truth")

ax = fig.add_subplot(122, projection='3d')
for i in range(len(labels)):
    if labels[i] == 0:
        ax.scatter(X[i][0], X[i][1], X[i][2], c='yellow')
    elif labels[i] == 1:
        ax.scatter(X[i][0], X[i][1], X[i][2], c='grey')
    elif labels[i] == 2:
        ax.scatter(X[i][0], X[i][1], X[i][2], c='brown')
ax.set_title("predict")
plt.savefig('D:\\AMCS_2022\\result\\result.png')
plt.show()


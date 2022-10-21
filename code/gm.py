from cProfile import label
from sklearn import datasets
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
# iris = datasets.load_iris()
iris = datasets.load_iris()
# print(iris)
X, y = datasets.load_iris(return_X_y=True)

gm=GaussianMixture(n_components=3,covariance_type='full',random_state=0).fit(X,y)
print(gm.means_)
print(gm.covariances_)
print(gm.weights_)

labels=gm.predict(X)
# X=pd.concat
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
for i in range(len(labels)):
    if labels[i]==0:
        ax.scatter(X[i][0],X[i][1],X[i][2],c='r')
    elif labels[i]==1:
        ax.scatter(X[i][0],X[i][1],X[i][2],c='blue')
    elif labels[i]==2:
        ax.scatter(X[i][0],X[i][1],X[i][2],c='g')

plt.show()

print(labels)
print("***")
print(X)
# plt.scatter()
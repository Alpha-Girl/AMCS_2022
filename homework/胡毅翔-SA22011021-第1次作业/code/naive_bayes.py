from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score, recall_score, precision_score
import pandas as pd
# load data
train_file = 'D:\\AMCS_2022\\data\\Bayesian_Dataset_train.csv'  # training set
df = pd.read_csv(train_file, header=None)
X_train = df.iloc[:, :-1]
y_train = df.iloc[:, -1]

test_file = 'D:\\AMCS_2022\\data\\Bayesian_Dataset_test.csv'  # test set
df = pd.read_csv(test_file, header=None)
X_test = df.iloc[:, :-1]
X_test_before_transform = X_test
y_test = df.iloc[:, -1]

X = pd.concat([X_test, X_train])

# data preprocessing

OrdEnc = OrdinalEncoder()
X = OrdEnc.fit_transform(X)
X = pd.DataFrame(X)
X_test = X.iloc[:len(X_test), :]
X_train = X.iloc[len(X_test):, :]

# naive bayes
# select Naive Bayes classifier
# and set parameters
# nb = BernoulliNB()  # Naive Bayes classifier for multivariate Bernoulli models.
# nb = GaussianNB(var_smoothing=1e-9)  # Gaussian Naive Bayes
nb = CategoricalNB(alpha=1.0,fit_prior=True)  # Naive Bayes classifier for categorical features.
# nb = ComplementNB()  # Complement Naive Bayes classifier
# nb = MultinomialNB()  # Naive Bayes classifier for multinomial models
y_pred = nb.fit(X_train, y_train).predict(X_test)

# print result statistics
print("positive label : ' >50K'")
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
print("Accuracy = %.4f" % (1-(y_test != y_pred).sum()/X_test.shape[0]))
print("Precision = %.4f" % (precision_score(y_test, y_pred, pos_label=' >50K')))
print("Recall = %.4f" % (recall_score(y_test, y_pred, pos_label=' >50K')))
print("F1_score = %.4f" % (f1_score(y_test, y_pred, pos_label=' >50K')))

# save prediction
y_pred = pd.Series(y_pred)
result = pd.concat([X_test_before_transform, y_test, y_pred], axis=1)
result.to_csv("D:\\AMCS_2022\\data\\Bayesian_Dataset_pred.csv", header=0)

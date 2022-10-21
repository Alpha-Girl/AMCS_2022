
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
test_file='D:\\AMCS_2022\\data\\Bayesian_Dataset_test.csv'
OrdEnc=OrdinalEncoder()
df=pd.read_csv(test_file,header=None)

X_test=df.iloc[:,:-1]
# print(len(X_test))
y_test=df.iloc[:,-1]
train_file='D:\\AMCS_2022\\data\\Bayesian_Dataset_train.csv'
df=pd.read_csv(train_file,header=None)
X_train=df.iloc[:,:-1]
y_train=df.iloc[:,-1]
X=pd.concat([X_test,X_train])

X=OrdEnc.fit_transform(X)
X=pd.DataFrame(X)
X_test=X.iloc[:len(X_test),:]
# print(X_test)
X_train=X.iloc[len(X_test):,:]
# print(X_train)
# print(y_test)
# print(y_train)
# X, y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
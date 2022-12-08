import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load data
train_file = 'D:\\AMCS_2022\\data\\kddcup99_train.csv'  # training set
df = pd.read_csv(train_file, header=None)
df[41]=df[41].apply(lambda x:0 if x=="normal." else 1)
X_train = df.iloc[:, :-1]
y_train = df.iloc[:, -1]

test_file = 'D:\\AMCS_2022\\data\\kddcup99_test.csv'  # test set
df = pd.read_csv(test_file, header=None)
df[41]=df[41].apply(lambda x:0 if x=="normal." else 1)
X_test = df.iloc[:, :-1]
y_test = df.iloc[:, -1]


X = pd.concat([X_test, X_train])

# data preprocessing

OrdEnc = OrdinalEncoder()
X = OrdEnc.fit_transform(X)
X = pd.DataFrame(X)
X_test = X.iloc[:len(X_test)]
X_train = X.iloc[len(X_test):]
# WIP

# initialization="uniform"
initialization="normal"
optimizer="SGD"
# optimizer="adam"
model = Sequential([
  Dense(41, activation='relu',kernel_initializer=initialization),
  Dense(36, activation='softmax',kernel_initializer=initialization),
  Dense(24, activation='relu',kernel_initializer=initialization),
  Dense(12, activation='relu',kernel_initializer=initialization),
  Dense(6, activation='softmax',kernel_initializer=initialization),
  Dense(1, activation='softmax',kernel_initializer=initialization),
])

model.compile(
  optimizer=optimizer,
  loss='binary_crossentropy',
  metrics=['binary_accuracy'],
)


history=model.fit(
  X_train, # training data
  y_train, # training targets
  epochs=3,
  batch_size=32,
  validation_data=(X_test,y_test)
)

model.summary()
print("initialization: ",initialization)
print("optimizer :",optimizer)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("loss_"+initialization+"_"+optimizer+"_rs2r2s"+".png")
plt.show()



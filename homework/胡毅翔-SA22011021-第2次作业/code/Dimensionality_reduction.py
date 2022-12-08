from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.decomposition import PCA
import torch.nn as nn
import matplotlib.pyplot as plt
import torch


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(41, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, n_components)
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_components, 8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 41),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# load data
train_file = 'D:\\AMCS_2022\\data\\kddcup99_train.csv'  # training set
df = pd.read_csv(train_file, header=None)
df[41] = df[41].apply(lambda x: 0 if x == "normal." else 1)
X_train = df.iloc[:, :-1]
y_train = df.iloc[:, -1]

test_file = 'D:\\AMCS_2022\\data\\kddcup99_test.csv'  # test set
df = pd.read_csv(test_file, header=None)
df[41] = df[41].apply(lambda x: 0 if x == "normal." else 1)
X_test = df.iloc[:, :-1]
y_test = df.iloc[:, -1]

X = pd.concat([X_test, X_train])

# data preprocessing

OrdEnc = OrdinalEncoder()
X = OrdEnc.fit_transform(X)
X = pd.DataFrame(X)

method = "pca"
# method = "autoencoder"
for n in [2, 3, 4]:
    n_components = n
    print(n_components)
    if method == "pca":
        pca = PCA(n_components=n_components, random_state=42)
        result = pca.fit_transform(X)

        X_test = result[:len(X_test)]
        X_train = result[len(X_test):]

        rf_clf = RandomForestClassifier(criterion="entropy")
        rf_clf.fit(X_train, y_train)

        y_predict = rf_clf.predict(X_test)
        print(accuracy_score(y_test, y_predict))
        X_test_d = pd.DataFrame(X_test)
        X_train_d = pd.DataFrame(X_train)
        if n == 2:
            fig = plt.figure(figsize=(30, 10))
            ax = fig.add_subplot(131)
            ax.scatter(X_train_d[:][0], X_train_d[:][1], c=y_train[:])
            ax.set_title("training set")
            bx = fig.add_subplot(132)
            bx.scatter(X_test_d[:][0], X_test_d[:][1], c=y_test[:])
            bx.set_title("truth on test set")
            cx = fig.add_subplot(133)
            cx.scatter(X_test_d[:][0], X_test_d[:][1], c=y_predict[:])
            cx.set_title("predict on test set")

            plt.savefig('D:\\AMCS_2022\\result\\pca2.png')

    elif method == "autoencoder":
        X_test = X.iloc[:len(X_test)]
        X_train = X.iloc[len(X_test):]

        Coder = AutoEncoder()
        EPOCH = 2
        BATCH_SIZE = 16
        LR = 0.005
        N_TEST_IMG = 5
        print(Coder)
        optimizer = torch.optim.Adam(Coder.parameters(), lr=LR)
        loss_func = nn.MSELoss()
        for epoch in range(EPOCH):
            for i in range(len(X_train)):
                tensor_t = (torch.from_numpy(X_train.iloc[i, :].values)).t()
                tensor_t = tensor_t.float()
                b_x = tensor_t
                b_y = tensor_t
                encoded, decoded = Coder(b_x)
                loss = loss_func(decoded, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        torch.save(Coder, 'AutoEncoder'+str(n_components)+'.pkl')
        print('________________________________________')
        print('finish training')

        Coder = torch.load('AutoEncoder'+str(n_components)+'.pkl')
        X_t = []
        for i in range(len(X_train)):
            tensor_t = (torch.from_numpy(X_train.iloc[i, :].values)).t()
            tensor_t = tensor_t.float()
            encoded, _ = Coder(tensor_t)
            X_t.append(encoded.detach().numpy())

        X_train_d = pd.DataFrame(X_t)
        print("[Start fitting...]")
        rf_clf = RandomForestClassifier(criterion="entropy")
        rf_clf.fit(X_train_d, y_train)
        print("[Finish fitting...]")

        X_te = []
        for i in range(len(X_test)):
            tensor_t = (torch.from_numpy(X_test.iloc[i, :].values)).t()
            tensor_t = tensor_t.float()
            encoded, _ = Coder(tensor_t)
            X_te.append(encoded.detach().numpy())
            if i % 1000000 == 0:
                print("processing data:", i/(1.0*len(X_test)))
        X_test_d = pd.DataFrame(X_te)
        y_predict = rf_clf.predict(X_test_d)
        print(accuracy_score(y_test, y_predict))

        if n == 2:
            fig = plt.figure(figsize=(30, 10))
            ax = fig.add_subplot(131)
            ax.scatter(X_train_d[:][0], X_train_d[:][1], c=y_train[:])
            ax.set_title("training set")
            bx = fig.add_subplot(132)
            bx.scatter(X_test_d[:][0], X_test_d[:][1], c=y_test[:])
            bx.set_title("truth on test set")
            cx = fig.add_subplot(133)
            cx.scatter(X_test_d[:][0], X_test_d[:][1], c=y_predict[:])
            cx.set_title("predict on test set")
    
            plt.savefig('D:\\AMCS_2022\\result\\auto2.png')

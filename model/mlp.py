""" Multilayer perceptron (MLP)

"""
# Author: xxx
# License: xxx

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MLPCell(nn.Module):
    def __init__(self, in_dim=100, hid_dim=32, out_dim=2):
        super(MLPCell, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

    def set_params(self):
        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim * 2),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(self.hid_dim * 2, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class MLP:

    def __init__(self, epochs=50, lr=1e-1, bs=64, in_dim=97, hid_dim=32, out_dim=2, random_state=42):
        """

        Parameters
        ----------
        epochs
        lr
        bs
        in_dim
        hid_dim
        out_dim
        random_state
        """
        self.model = MLPCell(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim)
        self.epochs = epochs
        self.lr = lr
        self.bs = bs
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.BCELoss()
        self.random_state = random_state

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.model, k, v)
            setattr(self, k, v)

        self.model.set_params()

    def get_params(self):
        for k, v in self.__dict__.items():
            print(f'{k}:{v}, ')
        print()

    def fit(self, X, y):
        """

        Parameters
        ----------
        X
        y

        Returns
        -------

        """
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        mean_train_losses = []
        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
        for epoch in range(self.epochs):

            train_losses = []
            train_loader = DataLoader(dataset=dataset, batch_size=self.bs, shuffle=True)
            for i, (xs, ys) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outs = self.model(xs)
                # outs, _ = torch.max(outs, axis=1)
                # mseloss and BCELoss
                ys = torch.nn.functional.one_hot(ys.type(torch.LongTensor)).type(torch.FloatTensor)
                # ys = ys.type(torch.LongTensor) # CrossEntropyLoss
                loss = self.loss_fn(outs, ys)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            mean_train_losses.append(np.mean(train_losses))
            print('epoch : {}, train loss : {:.4f}'.format(epoch + 1, np.mean(train_losses)))

    def _predict(self, X):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """

        self.model.eval()

        outs = []
        dataset = TensorDataset(torch.Tensor(X), )
        test_loader = DataLoader(dataset=dataset, batch_size=self.bs, shuffle=False)
        with torch.no_grad():
            for i, (xs,) in enumerate(test_loader):
                outs_ = self.model(xs)
                outs.extend(outs_.numpy().tolist())

        return np.asarray(outs)

    def predict_proba(self, X):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """
        # return predict X as '1' scores
        outs = self._predict(X)
        y_score = outs[:, 1]
        # print(np.quantile(y_score, q=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]))
        return y_score

    def predict(self, X):
        outs = self._predict(X)
        y_pred = np.argmax(outs, axis=1)

        return y_pred

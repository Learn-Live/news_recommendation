""" Model building and evaluation

"""
# Author: xxx
# License: xxx

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier

from model.mlp import MLP


class Recommender:

    def __init__(self, model_name='', model_params={}, random_state=42):
        """ Build and evaluate model

        Parameters
        ----------
        model_name
        model_params
        random_state
        """
        self.random_state = random_state

        if model_name == 'DecisionTreeClassifier':
            self.model = DecisionTreeClassifier(random_state=self.random_state)
        elif model_name == 'MLP':
            self.model = MLP(random_state=self.random_state)
        else:
            msg = f"{model_name}"
            raise NotImplementedError(msg)

        self.model.set_params(**model_params)
        print(self.model.get_params())

    def fit(self, X, y):
        """ fit a model on X, y

        Parameters
        ----------
        X
        y

        Returns
        -------

        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """ predict X

        Parameters
        ----------
        X

        Returns
        -------

        """
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def test(self, X, y):
        """ generate model's evaluation report

        Parameters
        ----------
        X
        y

        Returns
        -------

        """
        y_score = self.predict_proba(X)
        fpr, tpr, _ = roc_curve(y, y_score)
        auc_score = auc(fpr, tpr)
        print(f"auc: {auc_score:.4f}")

        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"acc: {acc:.4f}")
        cm = confusion_matrix(y, y_pred)
        print(cm)
        rp = classification_report(y, y_pred)
        print(rp)

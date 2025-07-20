from __future__ import annotations
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from datetime import datetime

from typing import Optional
from optuna.exceptions import TrialPruned


def score(clf, x, y):
    if not np.all(np.isin(y, [0, 1])):
        y = y * .5 + 0.5  # Convert -1, 1 to 0, 1
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        early_stopping_rounds: Optional[int] = None,
        bagging_temperature: float | int = 1.0,
        bootstrap_type = None,
        rsm: float | int = 1.0,
        quantization_type = None,
        nbins: int = 255,
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators
        self.early_stopping_rounds: int = early_stopping_rounds or np.inf
        
        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.history = defaultdict(list) # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: y * self.sigmoid(-y * z)

        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type

        self.quantization_type = quantization_type
        self.rsm = rsm
        self.nbins = nbins
        self._rsm_feature_indices_per_model: list = []
        self._quantization_bins: Optional[np.ndarray] = None
        self.feature_importances_ = None
        
    def _quantization_transform(self, X: np.ndarray) -> np.ndarray:
        if self.quantization_type is None or self._quantization_bins is None:
            return X
        X_ = X.copy()
        for i in range(X.shape[1]):
            X_[:, i] = np.digitize(X_[:, i], self._quantization_bins[:, i])
        return X_

    def _quantization_fit_transform(self, X: np.ndarray) -> np.ndarray:
        if self.quantization_type is None:
            return X
        if self.quantization_type == 'uniform':
            self._quantization_bins = np.linspace(np.min(X, axis=0), np.max(X, axis=0), self.nbins + 1)
            return self._quantization_transform(X)
        if self.quantization_type == 'quantile':
            self._quantization_bins = np.quantile(X, np.linspace(0, 1, self.nbins + 1), axis=0)
            return self._quantization_transform(X)

    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False, trial=None):

        train_predictions = np.zeros(y_train.shape[0])
        validation_predictions = np.zeros(y_val.shape[0]) if y_val is not None else None

        if not np.all(np.isin(y_train, [-1, 1])):
            y_train_ = y_train * 2 - 1
        else:
            y_train_ = y_train.copy()
        
        if y_val is not None:
            if not np.all(np.isin(y_val, [-1, 1])):
                y_val_ = y_val * 2 - 1
            else:
                y_val_ = y_val.copy()
        else: y_val_ = None

        time_start = datetime.now()
        patience = self.early_stopping_rounds
        best_score = -np.inf
        best_model_depth = 0

        X_train_quantized = self._quantization_fit_transform(X_train)
        X_val_quantized = self._quantization_transform(X_val) if X_val is not None else None

        if self.bootstrap_type is None:
            X_train_sample = X_train_quantized.copy()
            weights_sample = np.ones(X_train_sample.shape[0])

        for i in range(self.n_estimators):

            residuals = self.loss_derivative(y_train_, train_predictions)
            if self.bootstrap_type == 'Bernoulli':
                sample_indices_train = np.random.choice(X_train.shape[0], size=X_train.shape[0], replace=True)
                X_train_sample = X_train_quantized[sample_indices_train]
                residuals = residuals[sample_indices_train]
                weights_sample = np.ones(X_train_sample.shape[0])
            elif self.bootstrap_type == 'Bayesian':
                X_train_sample = X_train_quantized.copy()
                weights_sample = np.pow(-np.log(np.random.uniform(low=1e-9, high=1., size=X_train.shape[0])), self.bagging_temperature)
            
            X_val_sample = X_val_quantized.copy() if X_val_quantized is not None else None
            if self.rsm < 1:
                column_indices = np.random.choice(X_train.shape[1], size=int(X_train.shape[1]*self.rsm), replace=False)
                X_train_sample = X_train_sample[:, column_indices]
                X_val_sample = X_val_sample[:, column_indices] if X_val is not None else None
                self._rsm_feature_indices_per_model.append(column_indices)
            else: 
                self._rsm_feature_indices_per_model.append(np.arange(X_train.shape[1]))
            
            model = self.base_model_class(**self.base_model_params).fit(X_train_sample, residuals, sample_weight=weights_sample)
            self.models.append(model)

            gamma_step = self.find_optimal_gamma(y_train_, train_predictions, self.models[i].predict(
                X_train_quantized[:, column_indices] if self.rsm < 1 else X_train_quantized
            ))
            self.gammas.append(gamma_step)

            train_predictions += self.learning_rate * self.gammas[i] * self.models[i].predict(
                X_train_quantized[:, column_indices] if self.rsm < 1 else X_train_quantized
            )
            self.history['train_roc_auc'].append(score(self, X_train, y_train_))
            self.history['train_loss'].append(self.loss_fn(y_train_, train_predictions))

            if (X_val_sample is not None) and (y_val is not None):
                validation_predictions += self.learning_rate * self.gammas[i] * self.models[i].predict(
                    X_val_quantized[:, column_indices] if self.rsm < 1 else X_val_quantized
                )
                self.history['validation_roc_auc'].append(score(self, X_val, y_val))
                self.history['validation_loss'].append(self.loss_fn(y_val_, validation_predictions))
                if self.history['validation_roc_auc'][-1] > best_score:
                    best_score = self.history['validation_roc_auc'][-1]
                    best_model_depth = len(self.models)
                    patience = self.early_stopping_rounds
                else: patience -= 1

                if trial is not None:
                    trial.report(self.history['validation_roc_auc'][-1], i)
                    if trial.should_prune():
                        raise TrialPruned()
            
            if patience == 0:
                self.models = self.models[:best_model_depth]
                self.gammas = self.gammas[:best_model_depth]
                print(f'Early stopping at iteration {i}, best model depth: {len(self.models)}')
                break

            if i%10 == 0 and i > 0:
                print(f"{i}/{self.n_estimators}, ROC AUC: {self.history['train_roc_auc'][-1]:.4f}, Loss: {self.history['train_loss'][-1]:.4f}, Time: {(datetime.now()-time_start).total_seconds():.2f}")
                time_start = datetime.now()

        self.feature_importances_ = np.zeros(X_train.shape[1])
        for i, model in enumerate(self.models):
            if hasattr(model, 'feature_importances_'):
                self.feature_importances_[self._rsm_feature_indices_per_model[i]] += model.feature_importances_
        self.feature_importances_ /= len(self.models)
        self.feature_importances_ /= np.sum(self.feature_importances_) + 1e-9  # Normalize feature importance

        if plot:
            self.plot_history()
        
        return self
    
    def predict(self, X):
        return self.predict_proba(X)[:, 1] > .5

    def predict_proba(self, X):
        predictions = np.zeros((X.shape[0], 2))
        X_quantized = self._quantization_transform(X)
        for i, model in enumerate(self.models):
            X_sample = X_quantized[:, self._rsm_feature_indices_per_model[i]] if self._rsm_feature_indices_per_model else X_quantized
            predictions[:, 1] += self.learning_rate * self.gammas[i] * model.predict(X_sample)
        predictions[:, 1] = self.sigmoid(predictions[:, 1])
        predictions[:, 0] = 1 - predictions[:, 1]
        return predictions

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)
        
    def plot_history(self):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        plt.plot(self.history["train_roc_auc"], label="train_roc_auc")
        plt.plot(self.history["train_loss"], label="train_loss")

        plt.plot(self.history["validation_roc_auc"], label="validation_roc_auc")
        plt.plot(self.history["validation_loss"], label="validation_loss")

        plt.title("Boosting History")
        plt.legend()
        plt.show()
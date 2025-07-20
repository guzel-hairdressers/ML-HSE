from __future__ import annotations

from typing import List

import numpy as np

from descents import BaseDescent
from descents import get_descent


class LinearRegression:
    """
    Linear regression class
    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
        """
        :param descent_config: gradient descent config
        :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
        :param max_iter: stopping criterion for iterations (int)
        """
        self.descent: BaseDescent = get_descent(descent_config)
        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Fitting descent weights for x and y dataset
        :param x: features array
        :param y: targets array
        :return: self
        """
        descent_name = self.descent.__class__.__name__
        self.loss_history.append(self.descent.calc_loss(x, y))
        for i in range(self.max_iter):
            gradient = self.descent.calc_gradient(x, y)
            dw = self.descent.update_weights(gradient)
            self.loss_history.append(self.descent.calc_loss(x, y))
            
            if ((descent_name == 'Adam' or descent_name == 'AdamReg') and np.linalg.norm(dw) < 1e-10 and i>50)\
                or (descent_name != 'Adam' and descent_name != 'AdamReg' and np.linalg.norm(dw)**2 < self.tolerance)\
                or (np.any(np.isnan(dw))): break
                
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicting targets for x dataset
        :param x: features array
        :return: prediction: np.ndarray
        """
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating loss for x and y dataset
        :param x: features array
        :param y: targets array
        """
        return self.descent.calc_loss(x, y)

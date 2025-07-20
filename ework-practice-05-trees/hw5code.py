from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd



class DecisionTreeClassifier:
    def __init__(self, feature_types=None, max_depth=None, min_samples_split=None, min_samples_leaf=None, encode=True):

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth

        if min_samples_split is None:
            min_samples_split = 2
        if min_samples_leaf is None:
            min_samples_leaf = 1
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

        self._y_means = {}
        self._label_smoothing = 10
        self._encode = encode

    def find_best_split(self, X, y, min_samples_leaf=1):
        y = np.array(y)
        unique_features = np.unique(X)
        unique_features.sort()
        if unique_features.size <= 1:
            return (np.array([]), np.array([]), None, -np.inf)
        thresholds = (unique_features[:-1] + unique_features[1:]) / 2

        mask_l = np.array(X) < thresholds.reshape((-1,1))
        mask_r = 1 - mask_l
        n_l = mask_l.sum(axis=1)
        n_r = mask_r.sum(axis=1)
        p_l = np.divide((mask_l*y).sum(axis=1), n_l, out=np.zeros_like(n_l, dtype=float), where=(n_l!=0))
        p_r = np.divide((mask_r*y).sum(axis=1), n_r, out=np.zeros_like(n_r, dtype=float), where=(n_r!=0))

        gini_l = 2*p_l*(1-p_l)
        gini_r = 2*p_r*(1-p_r)
        ginis = -(gini_l*n_l + gini_r*n_r)/len(y)

        mask = (n_l > 0) & (n_r > 0) & (n_l >= min_samples_leaf) & (n_r >= min_samples_leaf)
        if mask.sum() == 0:
            return (np.array([]), np.array([]), None, -np.inf)

        valid_ginis = ginis[mask]
        valid_thresholds = thresholds[mask]
        gini_best = np.max(valid_ginis)
        threshold_best = valid_thresholds[np.argmax(valid_ginis)]

        return (thresholds, ginis, threshold_best, gini_best)

    def _mean_target_encoding_fit(self, X, y):
        X_cat = X.loc[:, self._feature_types == 'categorical']
        features_with_y = pd.concat([X_cat, y], axis=1)
        for feature in X_cat.columns:
            feature_y = features_with_y.groupby(feature)[y.name]
            self._y_means[feature] = (feature_y.sum() + self._label_smoothing*y.mean()) / (feature_y.size + self._label_smoothing)

        self._y_means['rare_or_unknown'] = y.mean()
        return self
    
    def _mean_target_encoding_transform(self, X):
        X_ = X.copy()
        for feature in X_.loc[:, self._feature_types == 'categorical'].columns:
            X_[feature] = X_[feature].map(self._y_means[feature])
            X_[feature].fillna(self._y_means['rare_or_unknown'], inplace=True)
        return X_

    def _fit_node(self, X, y, current_depth=0):
        if (np.unique(y).size == 1) or (len(y) < self._min_samples_split) or (self._max_depth is not None and current_depth >= self._max_depth):
            return {'type': 'terminal', 'prediction': Counter(y).most_common(1)[0][0]}
        X_ = X.copy()
        node = {}

        gini_best = -np.inf
        feature_best = None
        threshold_best = None
        split_best = None

        for feature in X_.columns:
            _, _, threshold, gini = self.find_best_split(X_[feature], y, self._min_samples_leaf)
            if gini > gini_best:
                gini_best = gini
                split_best = X_[feature] < threshold
                feature_best = feature
                threshold_best = threshold

        if split_best is None:
            return {'type': 'terminal', 'prediction': Counter(y).most_common(1)[0][0]}

        node['type'] = 'nonterminal'
        node['feature_split'] = feature_best
        node['threshold'] = threshold_best
        node['left_child'] = self._fit_node(X_.loc[split_best], y.loc[split_best], current_depth+1)
        node['right_child'] = self._fit_node(X_.loc[~split_best], y.loc[~split_best], current_depth+1)

        return node


    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['prediction']
            
        feature = node['feature_split']
        threshold = node['threshold']
        if x[feature] < threshold:
            return self._predict_node(x, node['left_child'])
        else:
            return self._predict_node(x, node['right_child'])
        
        

    def fit(self, X, y):
        if self._feature_types is None:
            self._feature_types = np.array(['real']*X.shape[1])
            self._feature_types[X.dtypes == 'object'] = 'categorical'

        X_ = X.copy()
        if self._encode:
            X_ = self._mean_target_encoding_fit(X_, y)._mean_target_encoding_transform(X_)

        self._tree = self._fit_node(X_, y)
        return self

    def predict(self, X):

        X_ = X.copy()
        if self._encode:
            X_ = self._mean_target_encoding_transform(X_)

        predicted = []
        for _, x in X_.iterrows():
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)




class LinearRegressionTree(DecisionTreeClassifier):

    def __init__(self, feature_types=None, max_depth=None, min_samples_split=None, min_samples_leaf=None, base_model=None, bins=10, encode=True):
        super().__init__(feature_types=feature_types, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, encode=encode)
        self.base_model = base_model or LinearRegression
        self.bins = bins

    def find_best_split(self, X, y, feature, min_samples_leaf=1):

        X_feature_unique = np.unique(X[feature])

        bin_size = max(1, X_feature_unique.size // self.bins)
        thresholds = X_feature_unique[bin_size: -bin_size: bin_size]

        losses = []

        for threshold in thresholds:
            mask_l = X[feature] < threshold
            mask_r = ~mask_l
            n_l = mask_l.sum()
            n_r = mask_r.sum()

            if n_l < min_samples_leaf or n_r < min_samples_leaf:
                losses.append(np.inf)
                continue

            predictions_l = self.base_model().fit(X.loc[mask_l], y[mask_l]).predict(X.loc[mask_l])
            predictions_r = self.base_model().fit(X.loc[mask_r], y[mask_r]).predict(X.loc[mask_r])
            mse_l = mean_squared_error(y[mask_l], predictions_l)
            mse_r = mean_squared_error(y[mask_r], predictions_r)
            loss = (mse_l * n_l + mse_r * n_r) / len(y)
            losses.append(loss)

        if len(losses) == 0 or np.all(np.isinf(losses)):
            return (np.array([]), np.array([]), None, np.inf)
        
        loss_best = np.min(losses)
        threshold_best = thresholds[np.argmin(losses)]

        return (thresholds, losses, threshold_best, loss_best)
    
    
    def _fit_node(self, X, y, current_depth=0):
        if (np.unique(y).size == 1) or (len(y) < self._min_samples_split) or (self._max_depth is not None and current_depth >= self._max_depth):
            return {'type': 'terminal', 'model': self.base_model().fit(X, y)}
        X_ = X.copy()
        node = {}

        loss_best = np.inf
        feature_best = None
        threshold_best = None
        split_best = None

        for feature in X_.columns:
            _, _, threshold, loss = self.find_best_split(X_, y, feature, self._min_samples_leaf)
            if loss < loss_best:
                loss_best = loss
                split_best = X_[feature] < threshold
                feature_best = feature
                threshold_best = threshold

        if split_best is None:
            return {'type': 'terminal', 'model': self.base_model().fit(X, y)}

        node['type'] = 'nonterminal'
        node['feature_split'] = feature_best
        node['threshold'] = threshold_best
        node['left_child'] = self._fit_node(X_.loc[split_best], y.loc[split_best], current_depth+1)
        node['right_child'] = self._fit_node(X_.loc[~split_best], y.loc[~split_best], current_depth+1)

        return node
    
    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['model'].predict(x.values.reshape(1,-1))[0]
            
        feature = node['feature_split']
        threshold = node['threshold']
        if x[feature] < threshold:
            return self._predict_node(x, node['left_child'])
        else:
            return self._predict_node(x, node['right_child'])

import numpy as np

from typing import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class FeatureCreatorPlaceholder(BaseEstimator, TransformerMixin):
    def __init__(self, n_features, new_dim, func: Callable = np.cos, use_PCA: bool = True, random_state=None):
        self.n_features = n_features
        self.new_dim = new_dim
        self.w = None
        self.b = None
        self.func = func
        self.pca = None
        self.use_PCA = use_PCA
        self.random_state = random_state

    def fit(self, X, y=None):
        if not self.use_PCA or self.new_dim == X.shape[1]:
            return self
        if self.new_dim > X.shape[1]:
            raise ValueError("new_dim must be less than or equal to the number of features in X")
        self.pca = PCA(n_components=self.new_dim, random_state=self.random_state).fit(X)
        return self

    def transform(self, X, y=None):
        if self.pca is not None:
            return self.pca.transform(X)
        return X.copy()

class RandomFeatureCreator(FeatureCreatorPlaceholder):
    def fit(self, X, y=None):
        if self.use_PCA:
            self.pca = PCA(n_components=self.new_dim, random_state=self.random_state).fit(X)
            X_pca = self.pca.transform(X)
        else: 
            X_pca = X.copy()
            self.new_dim = X_pca.shape[1]

        rs = np.random.RandomState(self.random_state)
        K = int(1e6)
        indices_l = rs.choice(X_pca.shape[0], K, replace=(X_pca.shape[0] < K))
        indices_r = rs.choice(X_pca.shape[0], K, replace=(X_pca.shape[0] < K))
        sigma2 = np.median(np.sum(np.pow(X_pca[indices_l] - X_pca[indices_r], 2), axis=-1))
        self.w = rs.normal(0, 1/np.sqrt(sigma2), (self.n_features, X_pca.shape[1]))
        self.b = rs.uniform(0, 2*np.pi, self.n_features)
        return self

    def transform(self, X, y=None):
        X_pca = super().transform(X, y)
        return self.func(X_pca @ self.w.T + self.b)


class OrthogonalRandomFeatureCreator(RandomFeatureCreator):
    def fit(self, X, y=None):
        if self.use_PCA:
            self.pca = PCA(n_components=self.new_dim, random_state=self.random_state).fit(X)
            X_pca = self.pca.transform(X)
        else: 
            X_pca = X.copy()
            self.new_dim = X_pca.shape[1]

        rs = np.random.RandomState(self.random_state)
        K = int(1e3)
        indices_l = rs.choice(X_pca.shape[0], K, replace=(X_pca.shape[0] < K))
        indices_r = rs.choice(X_pca.shape[0], K, replace=(X_pca.shape[0] < K))
        sigma2 = np.median(np.sum(np.pow(X_pca[indices_l] - X_pca[indices_r], 2), axis=-1)) + 1e-9
        weight_blocks = []
        for i in range(int(np.ceil(self.n_features / self.new_dim))):
            G = rs.normal(size=(self.new_dim, self.new_dim))
            Q, _ = np.linalg.qr(G)
            s = np.sqrt(rs.chisquare(self.new_dim, self.new_dim))
            weight_blocks.append(Q*s[:, None])
        self.w = np.vstack(weight_blocks)[:self.n_features] * 1/np.sqrt(sigma2)
        self.b = rs.uniform(0, 2*np.pi, self.n_features)
        return self

class RFFPipeline(BaseEstimator):
    """
    Пайплайн, делающий последовательно три шага:
        1. Применение PCA
        2. Применение RFF
        3. Применение классификатора
    """
    def __init__(
            self,
            n_features: int = 1000,
            new_dim: int = 50,
            use_PCA: bool = True,
            feature_creator_class=RandomFeatureCreator,
            classifier_class=LogisticRegression,
            classifier_params=None,
            func=np.cos,
            random_state=None
    ):
        """
        :param n_features: Количество признаков, генерируемых RFF
        :param new_dim: Количество признаков, до которых сжимает PCA
        :param use_PCA: Использовать ли PCA
        :param feature_creator_class: Класс, создающий признаки, по умолчанию заглушка
        :param classifier_class: Класс классификатора
        :param classifier_params: Параметры, которыми инициализируется классификатор
        :param func: Функция, которую получает feature_creator при инициализации.
                     Если не хотите, можете не использовать этот параметр.
        """
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        self.pipeline = None
        self.random_state = random_state
        if classifier_params is None:
            classifier_params = {}
        self.classifier = classifier_class(**classifier_params)
        self.feature_creator = feature_creator_class(
            n_features=self.n_features, new_dim=self.new_dim, func=func, use_PCA=self.use_PCA, random_state=self.random_state
        )
        

    def fit(self, X, y):
        pipeline_steps: list[tuple] = [
            ('feature_creator', self.feature_creator),
            ('classifier', self.classifier)
        ]
        self.pipeline = Pipeline(pipeline_steps).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)
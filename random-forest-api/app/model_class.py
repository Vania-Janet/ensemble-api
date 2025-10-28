"""
Implementación de SimpleRandomForest
"""

import numpy as np
from typing import List, Optional, Union
from sklearn.tree import DecisionTreeClassifier
from collections import Counter


class SimpleRandomForest:
    """
    Random Forest 'desde cero' usando DecisionTreeClassifier como base.
    - Bootstrap por árbol
    - Submuestreo aleatorio de features (max_features)
    - Votación mayoritaria en predict
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_features: Union[int, float, str, None] = "sqrt",
        max_depth: Optional[int] = None,
        criterion: str = "gini",
        random_state: Optional[int] = 42,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state
        self.trees: List[DecisionTreeClassifier] = []
        self._rng = np.random.default_rng(random_state)

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        idx = self._rng.integers(0, n, size=n)
        return X[idx], y[idx]

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.trees = []
        seeds = self._rng.integers(0, 10_000_000, size=self.n_estimators)
        for s in seeds:
            Xi, yi = self._bootstrap_sample(X, y)
            clf = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=int(s),
            )
            clf.fit(Xi, yi)
            self.trees.append(clf)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.column_stack([t.predict(X) for t in self.trees])
        maj = []
        for row in preds:
            c = Counter(row)
            maj.append(c.most_common(1)[0][0])
        return np.array(maj)

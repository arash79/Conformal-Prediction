import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from crepes import WrapClassifier
from typing import Optional


class ConformalClassifier:

    def __init__(
        self,
        base_estimator: Optional[RandomForestClassifier] = None,
        confidence: float = 0.95,
        random_state: Optional[int] = None,
    ):
        if base_estimator is None:
            base_estimator = RandomForestClassifier(
                n_estimators=500,
                n_jobs=-1,
                random_state=random_state,
            )
        self.confidence = confidence
        self.wrapper = WrapClassifier(base_estimator)
        self._is_calibrated = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.wrapper.fit(X_train, y_train)
        return self

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        self.wrapper.calibrate(X_cal, y_cal)
        self._is_calibrated = True
        return self

    def predict_set(self, X: np.ndarray):
        if not self._is_calibrated:
            raise RuntimeError("Model must be calibrated before prediction.")
        return self.wrapper.predict_set(X, confidence=self.confidence)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):

        if not self._is_calibrated:
            raise RuntimeError("Model must be calibrated before evaluation.")

        metrics = self.wrapper.evaluate(
            X_test,
            y_test,
            confidence=self.confidence,
            metrics=["error", "avg_c", "one_c"],
        )

        coverage = 1.0 - metrics["error"]
        avg_size = metrics["avg_c"]
        singleton_frac = metrics["one_c"]

        validity_score = 1.0 - abs(coverage - self.confidence)

        n_classes = self.wrapper.learner.n_classes_
        efficiency_score = 1.0 - (avg_size - 1.0) / (n_classes - 1.0)

        avg_size = metrics["avg_c"]
        one_rate = metrics["one_c"]

        efficiency_score_2 = one_rate / avg_size

        df = pd.DataFrame(
            {
                "confidence": [self.confidence],
                "avg_set_size": [metrics["avg_c"]],
                "singleton_fraction": [metrics["one_c"]],
                "coverage": [coverage],
                "efficiency": [efficiency_score],
                "efficiency_2": [efficiency_score_2],
                "validity_score": [1.0 - metrics["error"]],
                "validity_score_2": [validity_score],
            }
        )
        
        return df.reset_index(drop=True)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from crepes import WrapRegressor
from typing import Optional


class ConformalRegressor:

    def __init__(
        self,
        base_estimator: Optional[RandomForestRegressor] = None,
        confidence: float = 0.95,
        random_state: Optional[int] = None,
    ):
        if base_estimator is None:
            base_estimator = RandomForestRegressor(
                n_estimators=500,
                oob_score=True,  
                n_jobs=-1,
                random_state=random_state,
            )
        self.confidence = confidence
        self.wrapper = WrapRegressor(base_estimator)
        self._is_calibrated = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.wrapper.fit(X_train, y_train)
        return self

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        self.wrapper.calibrate(X_cal, y_cal)
        self._is_calibrated = True
        return self

    def predict_interval(self, X: np.ndarray):
        if not self._is_calibrated:
            raise RuntimeError("Model must be calibrated before prediction.")
        return self.wrapper.predict_int(X, confidence=self.confidence)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:

        if not self._is_calibrated:
            raise RuntimeError("Model must be calibrated before evaluation.")
        
        intervals = self.predict_interval(X_test)
        lower, upper = intervals[:, 0], intervals[:, 1]

        inside        = (y_test >= lower) & (y_test <= upper)
        coverage      = inside.mean()

        widths        = upper - lower
        width_mean    = widths.mean()
        width_median  = np.median(widths)
        width_sd = widths.std()

        validity_score   = 1.0 - abs(coverage - self.confidence)
        efficiency_score = 1.0 / width_mean

        y_pred = self.wrapper.predict(X_test)

        errors = y_test - y_pred
        mse    = np.mean(errors**2)
        rmse   = np.sqrt(mse)
        mae    = np.mean(np.abs(errors))

        ss_res = np.sum(errors**2)
        ss_tot = np.sum((y_test - y_test.mean())**2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        df = pd.DataFrame(
            {
                "confidence":   [self.confidence],
                "coverage":     [coverage],
                "width_mean":   [width_mean],
                "width_median": [width_median],
                "width_sd":     [width_sd],
                "validity_score":   [validity_score],
                "efficiency_score": [efficiency_score],
                "mse":          [mse],
                "rmse":         [rmse],
                "mae":          [mae],
                "r2":           [r2],
            }
        ).reset_index(drop=True)
    
        return df

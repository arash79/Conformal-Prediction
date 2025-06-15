import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler


class DataLoader:

    def __init__(
            self, 
            dataset: pd.DataFrame, 
            task: str = 'classification', 
            threshold: str = 'median',
            test_size: float = 0.2,
            calibration_size: float = 0.2,
            apply_dim_reduction: bool = False,
            dim_reduction_method: str = 'pca',  # or 'mutual_info'
            n_components: int = 10,             # for PCA
            top_k_features: int = 20            # for MI
            ):
        
        super().__init__()

        self.dataset = dataset.dropna(axis=0)
        self.task = task
        self.threshold = threshold
        self.test_size = test_size
        self.calibration_size = calibration_size
        self.apply_dim_reduction = apply_dim_reduction
        self.dim_reduction_method = dim_reduction_method.lower()
        self.n_components = n_components
        self.top_k_features = top_k_features

        self.__clean_up()
    
    def __clean_up(self):

        redundant_columns = ['SMILES', 'molecule', 'graph']
        self.dataset = self.dataset.drop(columns=redundant_columns, errors='ignore')

        self.dataset = self.dataset.rename(columns={'Ki (nM)': 'Ki'})
        self.dataset['pKi'] = -np.log10(self.dataset['Ki'] * 1e-9)
        self.dataset['target'] = self.dataset['pKi'].copy()

        if self.task == 'classification':
            
            if self.threshold.isalpha():
                threshold = getattr(np, self.threshold)(self.dataset['pKi'].values)
            else:
                threshold = float(self.threshold)

            self.dataset['target'] = (self.dataset['target'] >= threshold).astype(int)

        self.dataset = self.dataset.drop(columns=['Ki', 'pKi'])

    def __apply_dimensionality_reduction(self, X, y):
        
        if self.dim_reduction_method == 'pca':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=self.n_components)
            X_reduced = pca.fit_transform(X_scaled)
            return pd.DataFrame(X_reduced, index=X.index)

        elif self.dim_reduction_method == 'mutual_info':
            if self.task == 'classification':
                mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
                top_indices = np.argsort(mi_scores)[::-1][:self.top_k_features]
                selected_features = X.columns[top_indices]
                return X[selected_features]
            elif self.task == 'regression':
                mi_scores = mutual_info_regression(X, y, discrete_features='auto', random_state=42)
                top_indices = np.argsort(mi_scores)[::-1][:self.top_k_features]
                selected_features = X.columns[top_indices]
                return X[selected_features]

        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {self.dim_reduction_method}")

    def split(self):
        X, y = self.dataset.drop(columns=['target']), self.dataset['target']

        if self.apply_dim_reduction:
            X = self.__apply_dimensionality_reduction(X, y)

        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, 
            test_size=self.test_size + self.calibration_size, 
            random_state=42
        )
        
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_tmp, y_tmp, 
            test_size=self.test_size / (self.test_size + self.calibration_size), 
            random_state=42
        )

        return {
            'X_train': X_train, 'X_cal': X_cal, 'X_test': X_test,
            'y_train': y_train, 'y_cal': y_cal, 'y_test': y_test
        }

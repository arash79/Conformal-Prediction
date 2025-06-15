import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal


def mutual_info_to_target(
        df: pd.DataFrame,
        target: str,
        class_problem: Literal["binary", "multiclass", "regression"],
        maximum_features: int = 10,
        **mut_params
        ):

    if target not in df.columns.to_list():
        raise ValueError("Target column must be part of the provided DataFrame")

    if class_problem in ["binary", "multiclass"]:
        mi_scores = mutual_info_classif(
            X=df.drop(columns=[target]), y=df[target], **mut_params
        )
    else:
        mi_scores = mutual_info_regression(
            X=df.drop(columns=[target]), y=df[target], **mut_params
        )

    sorted_features = df.drop(columns=[target]).columns[np.argsort(-mi_scores)][:maximum_features]

    mi_scores_sorted = mi_scores[np.argsort(-mi_scores)][:maximum_features]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=mi_scores_sorted, y=sorted_features, ax=ax)

    ax.set_title("Mutual Information Scores with Target")
    ax.set_xlabel("Mutual Information Score")
    ax.set_ylabel("Features")

    ax.set_xlim([0, max(mi_scores_sorted) * 1.1])

    for i, (score, _feature) in enumerate(zip(mi_scores_sorted, sorted_features)):
        ax.annotate(
            f"{round(score, 2)}",
            xy=(score, i),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color="blue",
            fontweight="bold",
        )

    plt.tight_layout()  

    return fig

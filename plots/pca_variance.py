import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


def plot_pca_cumulative_variance(
        df: pd.DataFrame, scale_data: bool = True, 
        n_components: int = 10
        ):

    if scale_data:
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(df.copy())
    else:
        data_standardized = df.copy()

    pca = PCA(n_components=n_components)
    pca.fit(data_standardized)
    explained_variances = pca.explained_variance_ratio_

    individual_variances = explained_variances.tolist()

    cumulative_variances = np.cumsum(individual_variances)

    fig = plt.figure(figsize=(12, 7))
    plot_bar = plt.bar(
        range(1, n_components + 1),
        individual_variances,
        alpha=0.6,
        color="g",
        label="Individual Explained Variance",
    )

    plt.plot(
        range(1, n_components + 1),
        cumulative_variances,
        marker="o",
        linestyle="-",
        color="r",
        label="Cumulative Explained Variance",
    )

    for i, (bar, cum_val) in enumerate(zip(plot_bar, cumulative_variances)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{individual_variances[i]*100:.1f}%",
            ha="center",
            va="bottom",
        )
        plt.text(i + 1, cum_val, f"{cum_val*100:.1f}%", ha="center", va="bottom")

    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variance")
    plt.title("Explained Variance by Different Principal Components")
    plt.xticks(range(1, n_components + 1))
    plt.legend(loc="upper left")
    plt.ylim(0, 1.1) 
    plt.grid(True)

    return fig

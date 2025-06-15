import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def plot_pca(
        df: pd.DataFrame, target: str, 
        scale_data: bool = True
        ):
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    X = df.drop(columns=target).values
    y = df[target].values

    if scale_data:
        X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    comps = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_

    pca_df = pd.DataFrame(
        {'PC1': comps[:, 0], 'PC2': comps[:, 1], target: y},
        index=df.index
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    if len(pd.unique(y)) == 2:
        # binary classification
        palette = sns.color_palette("hls", len(pd.unique(y)))
        sns.scatterplot(
            data=pca_df,
            x='PC1', y='PC2',
            hue=target, palette=palette, ax=ax, s=60, edgecolor='w'
        )
    else:
        # regression or multiclass classification
        sns.scatterplot(
            data=pca_df,
            x='PC1', y='PC2',
            hue=target, palette='viridis', ax=ax, s=60, edgecolor='w', legend=None
        )

    ax.set_xlabel(f"PC1 ({evr[0]:.2%} var)")
    ax.set_ylabel(f"PC2 ({evr[1]:.2%} var)")
    ax.set_title(
        f"PCA Scatter (total explained: {(evr.sum()*100):.1f}%)",
        pad=15
    )
    sns.despine(trim=True)
    plt.tight_layout()

    return fig

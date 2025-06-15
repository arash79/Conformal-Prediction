from loader import DataLoader
from embeddings import *
from typing import List


def load_data(
        loader_type: str = 'descriptors',
        task: str = 'classification',
        fingerprint_type: str = None,
        threshold: str = 'median',
        test_size: float = 0.2,
        calibration_size: float = 0.2,
        descriptors: List[str] = None,
        embedding_size: int = 128,
        apply_dim_reduction: bool = False,
        dim_reduction_method: str = 'pca',  # or 'mutual_info'
        n_components: int = 10,             # for PCA
        top_k_features: int = 20            # for MI
        ):
    
    if loader_type == 'descriptors':
        embedding_generator = RDKDescriptors(descriptors)
    elif loader_type == 'graph':
        embedding_generator = GraphRepresentation(dimension_size=embedding_size)
    elif loader_type == 'fingerprints':
        embedding_generator = FingerPrints(
            fingerprint_type=fingerprint_type,
            fingerprint_size=embedding_size
        )
    
    initial_dataset = embedding_generator.embed()

    data_loader = DataLoader(
        dataset=initial_dataset,
        task=task,
        threshold=threshold,
        test_size=test_size,
        calibration_size=calibration_size,
        apply_dim_reduction=apply_dim_reduction,
        dim_reduction_method=dim_reduction_method,
        n_components=n_components,
        top_k_features=top_k_features
    )

    return data_loader

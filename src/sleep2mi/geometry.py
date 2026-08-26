from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def standardized_space(features: np.ndarray) -> np.ndarray:
    imputed = SimpleImputer(strategy="median").fit_transform(features)
    return StandardScaler().fit_transform(imputed)


def pca2_space(features: np.ndarray) -> tuple[np.ndarray, float]:
    standardized = standardized_space(features)
    n_components = min(2, standardized.shape[0] - 1, standardized.shape[1])
    if n_components < 1:
        raise ValueError("At least two participants and one feature are required")
    pca = PCA(n_components=n_components, svd_solver="full")
    projected = pca.fit_transform(standardized)
    return projected, float(pca.explained_variance_ratio_.sum())


def geometry_ratio(coordinates: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must contain both binary groups 0 and 1")
    low = coordinates[labels == 0]
    high = coordinates[labels == 1]
    low_center = low.mean(axis=0)
    high_center = high.mean(axis=0)
    between = float(np.linalg.norm(high_center - low_center))
    within = np.concatenate(
        [
            np.linalg.norm(low - low_center, axis=1),
            np.linalg.norm(high - high_center, axis=1),
        ]
    )
    return between / (float(within.mean()) + 1e-12)


def permutation_test(
    coordinates: np.ndarray,
    labels: np.ndarray,
    *,
    n_permutations: int = 5000,
    seed: int = 67,
) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    observed = geometry_ratio(coordinates, labels)
    null = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        null[index] = geometry_ratio(coordinates, rng.permutation(labels))
    p_value = (1.0 + float(np.sum(null >= observed))) / (n_permutations + 1.0)
    return p_value, null


def stratified_bootstrap_interval(
    coordinates: np.ndarray,
    labels: np.ndarray,
    *,
    n_bootstrap: int = 5000,
    seed: int = 67,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    low = np.flatnonzero(labels == 0)
    high = np.flatnonzero(labels == 1)
    values = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = np.concatenate(
            [
                rng.choice(low, size=len(low), replace=True),
                rng.choice(high, size=len(high), replace=True),
            ]
        )
        values[index] = geometry_ratio(coordinates[sample], labels[sample])
    lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


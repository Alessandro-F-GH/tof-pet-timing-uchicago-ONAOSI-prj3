"""Standard PPCA conditional-mean imputation utilities (no boundary correction)."""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA


def positions(length, mask_len, stride):
    if not 0 < mask_len < length or stride < 1:
        raise ValueError("Invalid mask width or stride")
    starts = list(range(0, length - mask_len + 1, stride))
    if starts[-1] != length - mask_len:
        starts.append(length - mask_len)
    return starts


class ProbabilisticPCA:
    """Full-waveform PPCA: low-rank Gaussian covariance plus isotropic noise."""
    def __init__(self, components, noise_floor=1e-10):
        self.components = int(components)
        self.noise_floor = float(noise_floor)

    def fit(self, data):
        data = np.asarray(data, dtype=np.float64)
        n, length = data.shape
        if not 1 <= self.components < min(n, length):
            raise ValueError("PPCA components must be less than samples and events")
        pca = PCA(n_components=self.components, svd_solver="randomized", random_state=42)
        pca.fit(data)
        self.mean = np.asarray(pca.mean_, dtype=np.float64)
        eigenvalues = pca.explained_variance_
        self.sigma2 = max(float(pca.noise_variance_), self.noise_floor)
        # PPCA maximum-likelihood loading matrix (rotation fixed to PCA axes).
        scales = np.sqrt(np.maximum(eigenvalues - self.sigma2, 0.))
        self.W = np.ascontiguousarray(pca.components_.T * scales)
        self.explained_variance_ratio = float(pca.explained_variance_ratio_.sum())
        return self

    def prepare_mask(self, start, width):
        """Cache the mask-specific posterior and predictive covariance factors."""
        stop = start + width
        if start < 0 or stop > len(self.mean):
            raise ValueError("Invalid mask")
        observed = np.ones(len(self.mean), dtype=bool)
        observed[start:stop] = False
        W_o, W_m = self.W[observed], self.W[start:stop]
        precision = np.eye(self.components) + W_o.T @ W_o / self.sigma2
        factor = np.linalg.cholesky(precision)
        # Missing mean = mu_M + (x_O - mu_O) @ coefficient_map
        # Solve with multiple RHS once, not once per event.
        coefficient_map = np.linalg.solve(
            factor.T, np.linalg.solve(factor, W_o.T / self.sigma2)
        ).T @ W_m.T
        # Integrate out z: x_M|x_O ~ N(mean_M, W_M Cov(z|x_O) W_M^T + sigma²I)
        latent_to_missing = np.linalg.solve(factor, W_m.T)
        covariance = latent_to_missing.T @ latent_to_missing
        covariance.flat[::width + 1] += self.sigma2
        sampling_factor = np.linalg.cholesky(covariance)
        return observed, coefficient_map, sampling_factor

    def conditional_mean_batch(self, waveforms, start, width, prepared=None):
        """Compute conditional means without generating stochastic draws."""
        observed, coefficient_map, _ = (
            prepared if prepared is not None else self.prepare_mask(start, width)
        )
        batch = np.asarray(waveforms, dtype=np.float64)
        return self.mean[start:start + width] + (
            batch[:, observed] - self.mean[observed]
        ) @ coefficient_map

    def condition_batch(self, waveforms, start, width, samples, rng, prepared=None):
        """Batch means and exact Gaussian conditional draws, shape [events, draws, width]."""
        if samples < 1:
            raise ValueError("samples must be >= 1")
        prepared = prepared if prepared is not None else self.prepare_mask(start, width)
        _, _, sampling_factor = prepared
        batch = np.asarray(waveforms, dtype=np.float64)
        mean = self.conditional_mean_batch(batch, start, width, prepared=prepared)
        random = rng.standard_normal((len(batch), samples, width))
        draws = mean[:, None, :] + random @ sampling_factor.T
        return mean, draws

    def condition(self, waveform, start, width, samples, rng):
        mean, draws = self.condition_batch(
            np.asarray(waveform)[None], start, width, samples, rng
        )
        return mean[0], draws[0]


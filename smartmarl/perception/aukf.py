"""Adaptive Unscented Kalman Filter with online measurement noise adaptation."""

from __future__ import annotations

import numpy as np


class AdaptiveUKF:
    """
    Adaptive UKF for traffic lane state estimation.

    State: [queue_length, vehicle_count, mean_speed, occupancy]
    Observation: same 4-D representation from fused camera/radar measurements.
    """

    def __init__(
        self,
        state_dim: int = 4,
        obs_dim: int = 4,
        beta: float = 0.02,
        alpha: float = 1e-3,
        kappa: float = 0.0,
    ) -> None:
        self.n = int(state_dim)
        self.m = int(obs_dim)
        if self.n != 4 or self.m != 4:
            raise ValueError("AdaptiveUKF in SmartMARL expects state_dim=4 and obs_dim=4")

        self.beta_adapt = float(beta)
        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n

        self.H = np.eye(self.m, self.n, dtype=np.float64)
        self.Q = np.diag([0.01] * self.n).astype(np.float64)

        self._x0 = np.zeros(self.n, dtype=np.float64)
        self._P0 = np.eye(self.n, dtype=np.float64)
        self._R0 = np.diag([0.5] * self.m).astype(np.float64)

        self.x = self._x0.copy()
        self.P = self._P0.copy()
        self.R = self._R0.copy()

        self.wm, self.wc = self._compute_weights()
        self._sigma2_r = np.diag(self.R).copy()

    @property
    def sigma2_r(self) -> np.ndarray:
        return self._sigma2_r.copy()

    def reset(self) -> None:
        self.x = self._x0.copy()
        self.P = self._P0.copy()
        self.R = self._R0.copy()
        self._sigma2_r = np.diag(self.R).copy()

    def _compute_weights(self) -> tuple[np.ndarray, np.ndarray]:
        n_sigma = 2 * self.n + 1
        wm = np.full(n_sigma, 1.0 / (2.0 * (self.n + self.lambda_)), dtype=np.float64)
        wc = wm.copy()

        wm[0] = self.lambda_ / (self.n + self.lambda_)
        wc[0] = wm[0] + (1.0 - self.alpha**2 + 2.0)
        return wm, wc

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        sigma_points = np.zeros((2 * self.n + 1, self.n), dtype=np.float64)
        sigma_points[0] = x

        scaled = (self.n + self.lambda_) * self._make_psd(P)
        jitter = 1e-8
        sqrtm = None
        for _ in range(5):
            try:
                sqrtm = np.linalg.cholesky(scaled + jitter * np.eye(self.n))
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        if sqrtm is None:
            eigvals, eigvecs = np.linalg.eigh(self._make_psd(scaled + 1e-3 * np.eye(self.n)))
            sqrtm = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 1e-10, None)))

        for i in range(self.n):
            sigma_points[i + 1] = x + sqrtm[:, i]
            sigma_points[self.n + i + 1] = x - sqrtm[:, i]
        return sigma_points

    @staticmethod
    def _process_model(x: np.ndarray) -> np.ndarray:
        # Constant-velocity approximation is represented as identity in this
        # compact 4-D state; evolution is captured through process noise.
        return x.copy()

    def _predict(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sigma_pts = self._sigma_points(self.x, self.P)
        sigma_pred = np.array([self._process_model(sp) for sp in sigma_pts])

        x_pred = np.sum(self.wm[:, None] * sigma_pred, axis=0)
        P_pred = self.Q.copy()
        for i in range(sigma_pred.shape[0]):
            dx = sigma_pred[i] - x_pred
            P_pred += self.wc[i] * np.outer(dx, dx)

        return sigma_pred, x_pred, self._make_psd(P_pred)

    @staticmethod
    def _make_psd(matrix: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
        sym = 0.5 * (matrix + matrix.T)
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals = np.clip(eigvals, min_eig, None)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _update_measurement_noise(self, innovation: np.ndarray, P_pred: np.ndarray) -> None:
        # Equation (3): R_k = (1-beta) R_{k-1} + beta (yy^T - HPH^T)
        target = np.outer(innovation, innovation) - self.H @ P_pred @ self.H.T
        self.R = (1.0 - self.beta_adapt) * self.R + self.beta_adapt * target
        self.R = 0.5 * (self.R + self.R.T)

        diag = np.clip(np.diag(self.R), 1e-6, None)
        self.R = self.R - np.diag(np.diag(self.R)) + np.diag(diag)
        self._sigma2_r = diag.copy()

    def update(self, z_camera: np.ndarray, z_radar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z_camera = np.asarray(z_camera, dtype=np.float64).reshape(self.m)
        z_radar = np.asarray(z_radar, dtype=np.float64).reshape(self.m)

        sigma_pred, x_pred, P_pred = self._predict()

        z_sigma = np.array([self.H @ sp for sp in sigma_pred])
        z_pred = np.sum(self.wm[:, None] * z_sigma, axis=0)

        P_zz = np.zeros((self.m, self.m), dtype=np.float64)
        P_xz = np.zeros((self.n, self.m), dtype=np.float64)
        for i in range(z_sigma.shape[0]):
            dz = z_sigma[i] - z_pred
            dx = sigma_pred[i] - x_pred
            P_zz += self.wc[i] * np.outer(dz, dz)
            P_xz += self.wc[i] * np.outer(dx, dz)

        z_k = 0.5 * (z_camera + z_radar)
        innovation = z_k - self.H @ x_pred

        self._update_measurement_noise(innovation, P_pred)

        S = P_zz + self.R
        K = P_xz @ np.linalg.pinv(S)

        self.x = x_pred + K @ (z_k - z_pred)
        self.P = P_pred - K @ S @ K.T
        self.P = self._make_psd(self.P)

        self.x[0] = max(self.x[0], 0.0)
        self.x[1] = max(self.x[1], 0.0)
        self.x[2] = np.clip(self.x[2], 0.0, 30.0)
        self.x[3] = np.clip(self.x[3], 0.0, 1.0)

        return self.x.copy(), self.sigma2_r

"""Temporal Fusion Transformer predictor for multi-horizon financial forecasting.

Provides a wrapper that attempts to use ``pytorch-forecasting``'s
TemporalFusionTransformer when available, falling back to a lightweight custom
implementation built with standard PyTorch when it is not.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.core.exceptions import (
    ModelError,
    ModelLoadError,
    ModelPredictionError,
    ModelTrainingError,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict[str, Any] = {
    "hidden_size": 64,
    "attention_head_size": 4,
    "num_attention_layers": 2,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "max_epochs": 50,
    "batch_size": 64,
    "patience": 8,
    "horizons": [1, 2, 5, 10, 21],
    "gradient_clip_val": 0.1,
    "sequence_length": 60,
    "validation_split": 0.1,
}

# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "values"):
        return np.asarray(x.values, dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Custom TFT components (fallback when pytorch-forecasting is unavailable)
# ---------------------------------------------------------------------------


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class _GatedLinearUnit(nn.Module):
    """GLU-based gating mechanism used for variable selection."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) * torch.sigmoid(self.gate(x))


class _VariableSelectionNetwork(nn.Module):
    """Simplified variable selection via learned feature gating.

    Produces per-feature weights that indicate how much each input
    variable contributes to the model.
    """

    def __init__(self, num_features: int, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_features = num_features
        self.flattened_grn = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_features),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.feature_transforms = nn.ModuleList(
            [_GatedLinearUnit(1, hidden_size) for _ in range(num_features)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, num_features)

        Returns:
            transformed: (batch, seq_len, hidden_size) — weighted sum of
                per-feature transformations.
            weights: (batch, num_features) — variable selection weights
                averaged over sequence positions.
        """
        # Compute selection weights from the raw features
        # Use mean-pooled representation for the weight computation
        flat = x.mean(dim=1)  # (B, F)
        raw_weights = self.flattened_grn(flat)  # (B, F)
        weights = self.softmax(raw_weights)     # (B, F)

        # Transform each feature independently
        B, S, F = x.shape
        transformed_features = []
        for i in range(F):
            feat_i = x[:, :, i : i + 1]  # (B, S, 1)
            transformed_features.append(self.feature_transforms[i](feat_i))  # (B, S, H)

        stacked = torch.stack(transformed_features, dim=2)  # (B, S, F, H)
        # Weight and sum
        w = weights.unsqueeze(1).unsqueeze(-1)  # (B, 1, F, 1)
        out = (stacked * w).sum(dim=2)          # (B, S, H)

        return out, weights


class _CustomTFTNetwork(nn.Module):
    """Lightweight Temporal Fusion Transformer built with standard PyTorch.

    Architecture:
    - Variable Selection Network (gating)
    - Positional Encoding
    - Stack of multi-head self-attention encoder layers
    - Per-horizon linear decoder heads (direction + returns per horizon)
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        attention_head_size: int = 4,
        num_attention_layers: int = 2,
        dropout: float = 0.1,
        horizons: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.horizons = horizons or [1, 2, 5, 10, 21]

        # Variable selection
        self.vsn = _VariableSelectionNetwork(num_features, hidden_size, dropout)

        # Positional encoding
        self.pos_enc = _PositionalEncoding(hidden_size, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_head_size,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_attention_layers)
        self.encoder_norm = nn.LayerNorm(hidden_size)

        # Per-horizon decoder heads
        self.direction_heads = nn.ModuleDict()
        self.return_heads = nn.ModuleDict()
        for h in self.horizons:
            key = str(h)
            self.direction_heads[key] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
            )
            self.return_heads[key] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, num_features)

        Returns:
            directions: dict mapping horizon str -> (batch, 1) probabilities.
            returns: dict mapping horizon str -> (batch, 1) return predictions.
            vsn_weights: (batch, num_features) variable selection weights.
        """
        selected, vsn_weights = self.vsn(x)    # (B, S, H), (B, F)
        encoded = self.pos_enc(selected)        # (B, S, H)
        encoded = self.encoder(encoded)         # (B, S, H)
        encoded = self.encoder_norm(encoded)

        # Aggregate: use last time-step as context
        context = encoded[:, -1, :]  # (B, H)

        directions: dict[str, torch.Tensor] = {}
        returns: dict[str, torch.Tensor] = {}
        for h in self.horizons:
            key = str(h)
            directions[key] = self.direction_heads[key](context)
            returns[key] = self.return_heads[key](context)

        return directions, returns, vsn_weights


# ---------------------------------------------------------------------------
# TFTPredictor (public interface)
# ---------------------------------------------------------------------------


class TFTPredictor:
    """Temporal Fusion Transformer predictor implementing the BasePredictor protocol.

    Supports multi-horizon forecasting: for each configured horizon, the model
    predicts both a directional probability and an expected return.

    Falls back to a custom lightweight transformer when ``pytorch-forecasting``
    is not installed.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = {**_DEFAULT_PARAMS, **(params or {})}
        self.device = _get_device()
        self.model: _CustomTFTNetwork | None = None
        self._fitted = False
        self._num_features: int | None = None
        self._feature_names: list[str] | None = None
        self._vsn_weights: np.ndarray | None = None
        self._best_val_loss: float = float("inf")
        self._use_pytorch_forecasting = False

        logger.info(
            "tft_predictor.init",
            device=str(self.device),
            horizons=self.params["horizons"],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted or self.model is None:
            raise ModelPredictionError(
                "Model is not fitted. Call fit() first.",
                details={"fitted": self._fitted},
            )

    def _validate_input(self, X: Any, *, fitting: bool = False) -> np.ndarray:
        arr = _to_numpy(X)
        if arr.ndim == 1:
            raise ModelError("Input must be at least 2-dimensional.", details={"shape": arr.shape})
        if not fitting and self._num_features is not None:
            actual = arr.shape[-1]
            if actual != self._num_features:
                raise ModelError(
                    f"Feature count mismatch: expected {self._num_features}, got {actual}.",
                )
        return arr

    def _build_sequences(self, X: np.ndarray) -> np.ndarray:
        seq_len = self.params["sequence_length"]
        if X.ndim == 3:
            return X
        n_samples, n_features = X.shape
        if n_samples < seq_len:
            raise ModelError(
                f"Not enough samples ({n_samples}) for sequence_length {seq_len}.",
            )
        sequences = np.lib.stride_tricks.sliding_window_view(X, seq_len, axis=0)
        return sequences.transpose(0, 2, 1).astype(np.float32)

    def _create_model(self, num_features: int) -> _CustomTFTNetwork:
        model = _CustomTFTNetwork(
            num_features=num_features,
            hidden_size=self.params["hidden_size"],
            attention_head_size=self.params["attention_head_size"],
            num_attention_layers=self.params.get("num_attention_layers", 2),
            dropout=self.params["dropout"],
            horizons=self.params["horizons"],
        )
        return model.to(self.device)

    def _try_pytorch_forecasting(self) -> bool:
        """Attempt to import pytorch-forecasting. Returns True if available."""
        try:
            import pytorch_forecasting  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any, **kwargs: Any) -> None:
        """Train the TFT model.

        Args:
            X: Feature matrix — (n_samples, n_features) or (n_samples, seq_len, n_features).
            y: Target array/DataFrame. For multi-horizon, expects shape
               (n_samples, n_horizons * 2) with pairs of
               (direction_h, return_h) for each horizon, or a dict
               ``{horizon: DataFrame[direction, returns]}``.
               Alternatively, a simple (n_samples, 2) with direction + return
               that will be shared across horizons.
            **kwargs: Optional ``validation_data=(X_val, y_val)``.
        """
        log = logger.bind(phase="fit")
        horizons = self.params["horizons"]
        n_horizons = len(horizons)

        # Check for pytorch-forecasting availability
        if self._try_pytorch_forecasting():
            log.info("pytorch_forecasting.available", note="falling back to custom TFT for consistency")
        else:
            log.info("pytorch_forecasting.unavailable", note="using custom TFT implementation")

        try:
            if hasattr(X, "columns"):
                self._feature_names = list(X.columns)

            X_arr = self._validate_input(X, fitting=True)
            y_arr = _to_numpy(y)

            X_seq = self._build_sequences(X_arr)
            seq_len = self.params["sequence_length"]

            # Align targets
            if X_arr.ndim == 2:
                y_arr = y_arr[seq_len - 1:]

            if y_arr.shape[0] != X_seq.shape[0]:
                raise ModelTrainingError(
                    "X and y length mismatch after sequencing.",
                    details={"X_sequences": X_seq.shape[0], "y_rows": y_arr.shape[0]},
                )

            # Parse targets into per-horizon direction + return arrays
            # Accepted formats:
            #   (n, 2): single direction + return -> broadcast to all horizons
            #   (n, 2*H): interleaved direction_h, return_h for each horizon
            y_directions: dict[str, np.ndarray] = {}
            y_returns: dict[str, np.ndarray] = {}

            if y_arr.ndim == 1 or (y_arr.ndim == 2 and y_arr.shape[1] == 1):
                # Single column — treat as direction only
                flat = y_arr.ravel()
                for h in horizons:
                    y_directions[str(h)] = flat
                    y_returns[str(h)] = np.zeros_like(flat)
            elif y_arr.shape[1] == 2:
                for h in horizons:
                    y_directions[str(h)] = y_arr[:, 0]
                    y_returns[str(h)] = y_arr[:, 1]
            elif y_arr.shape[1] == 2 * n_horizons:
                for idx, h in enumerate(horizons):
                    y_directions[str(h)] = y_arr[:, idx * 2]
                    y_returns[str(h)] = y_arr[:, idx * 2 + 1]
            else:
                raise ModelTrainingError(
                    f"Unexpected y shape {y_arr.shape}. Expected (n, 2) or (n, {2 * n_horizons}).",
                )

            self._num_features = X_seq.shape[2]

            # --- Validation split ------------------------------------
            val_split = self.params["validation_split"]
            val_data = kwargs.get("validation_data")

            if val_data is not None:
                X_val_seq = self._build_sequences(self._validate_input(val_data[0], fitting=True))
                y_val_arr = _to_numpy(val_data[1])
                if np.ndim(val_data[0]) == 2 if not hasattr(val_data[0], "ndim") else val_data[0].ndim == 2:
                    y_val_arr = y_val_arr[seq_len - 1:]
                # Parse val targets the same way
                y_val_dirs: dict[str, np.ndarray] = {}
                y_val_rets: dict[str, np.ndarray] = {}
                if y_val_arr.shape[1] == 2:
                    for h in horizons:
                        y_val_dirs[str(h)] = y_val_arr[:, 0]
                        y_val_rets[str(h)] = y_val_arr[:, 1]
                elif y_val_arr.shape[1] == 2 * n_horizons:
                    for idx, h in enumerate(horizons):
                        y_val_dirs[str(h)] = y_val_arr[:, idx * 2]
                        y_val_rets[str(h)] = y_val_arr[:, idx * 2 + 1]
            else:
                n = X_seq.shape[0]
                split_idx = int(n * (1 - val_split))
                X_val_seq = X_seq[split_idx:]
                X_seq = X_seq[:split_idx]

                y_val_dirs = {}
                y_val_rets = {}
                for key in y_directions:
                    y_val_dirs[key] = y_directions[key][split_idx:]
                    y_val_rets[key] = y_returns[key][split_idx:]
                    y_directions[key] = y_directions[key][:split_idx]
                    y_returns[key] = y_returns[key][:split_idx]

            # --- Build tensors ---------------------------------------
            X_train_t = torch.tensor(X_seq, dtype=torch.float32)
            X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)

            # Flatten all horizon targets into a single tensor for the dataset
            # Order: dir_h0, ret_h0, dir_h1, ret_h1, ...
            train_targets = []
            val_targets = []
            for h in horizons:
                key = str(h)
                train_targets.extend([
                    torch.tensor(y_directions[key], dtype=torch.float32),
                    torch.tensor(y_returns[key], dtype=torch.float32),
                ])
                val_targets.extend([
                    torch.tensor(y_val_dirs[key], dtype=torch.float32),
                    torch.tensor(y_val_rets[key], dtype=torch.float32),
                ])

            y_train_t = torch.stack(train_targets, dim=1)  # (N, 2*H)
            y_val_t = torch.stack(val_targets, dim=1)

            batch_size = self.params["batch_size"]
            train_loader = DataLoader(
                TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False,
            )

            # --- Model, optimizer ------------------------------------
            self.model = self._create_model(self._num_features)
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.params["learning_rate"],
                weight_decay=self.params["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6,
            )

            bce_fn = nn.BCELoss()
            mse_fn = nn.MSELoss()
            clip_val = self.params["gradient_clip_val"]

            best_val_loss = float("inf")
            best_state: dict[str, Any] | None = None
            epochs_no_improve = 0
            patience = self.params["patience"]
            max_epochs = self.params["max_epochs"]

            # --- Training loop ---------------------------------------
            for epoch in range(1, max_epochs + 1):
                self.model.train()
                train_loss_accum = 0.0
                n_batches = 0

                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    dirs, rets, _ = self.model(batch_x)

                    loss = torch.tensor(0.0, device=self.device)
                    for idx, h in enumerate(horizons):
                        key = str(h)
                        target_dir = batch_y[:, idx * 2].unsqueeze(1)
                        target_ret = batch_y[:, idx * 2 + 1].unsqueeze(1)
                        loss = loss + bce_fn(dirs[key], target_dir) + mse_fn(rets[key], target_ret)

                    loss = loss / n_horizons  # average across horizons

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_val)
                    optimizer.step()

                    train_loss_accum += loss.item()
                    n_batches += 1

                avg_train_loss = train_loss_accum / max(n_batches, 1)

                # --- Validation --------------------------------------
                self.model.eval()
                val_loss_accum = 0.0
                n_val = 0
                vsn_weights_accum: torch.Tensor | None = None

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        dirs, rets, vsn_w = self.model(batch_x)

                        v_loss = torch.tensor(0.0, device=self.device)
                        for idx, h in enumerate(horizons):
                            key = str(h)
                            target_dir = batch_y[:, idx * 2].unsqueeze(1)
                            target_ret = batch_y[:, idx * 2 + 1].unsqueeze(1)
                            v_loss = v_loss + bce_fn(dirs[key], target_dir) + mse_fn(rets[key], target_ret)
                        v_loss = v_loss / n_horizons

                        val_loss_accum += v_loss.item()
                        n_val += 1

                        # Accumulate VSN weights for feature importance
                        if vsn_weights_accum is None:
                            vsn_weights_accum = vsn_w.sum(dim=0)
                        else:
                            vsn_weights_accum = vsn_weights_accum + vsn_w.sum(dim=0)

                avg_val_loss = val_loss_accum / max(n_val, 1)
                scheduler.step(avg_val_loss)

                if epoch % 5 == 1 or epoch == max_epochs:
                    log.info(
                        "training.epoch",
                        epoch=epoch,
                        train_loss=round(avg_train_loss, 6),
                        val_loss=round(avg_val_loss, 6),
                        lr=optimizer.param_groups[0]["lr"],
                    )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                    # Store VSN weights from best epoch
                    if vsn_weights_accum is not None:
                        self._vsn_weights = vsn_weights_accum.cpu().numpy()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    log.info("training.early_stop", epoch=epoch, best_val_loss=round(best_val_loss, 6))
                    break

            if best_state is not None:
                self.model.load_state_dict(best_state)

            self._best_val_loss = best_val_loss
            self._fitted = True

            log.info(
                "training.complete",
                best_val_loss=round(best_val_loss, 6),
                num_features=self._num_features,
            )

        except (ModelTrainingError, ModelError):
            raise
        except Exception as exc:
            raise ModelTrainingError(
                f"Training failed: {exc}",
                details={"error": str(exc)},
            ) from exc

    def predict(self, X: Any) -> dict[int, dict[str, np.ndarray]]:
        """Generate multi-horizon predictions.

        Args:
            X: Feature matrix.

        Returns:
            Dictionary mapping horizon ``int`` to a dict with keys
            ``direction`` (probability) and ``returns`` (predicted return).
        """
        self._check_fitted()
        assert self.model is not None

        try:
            X_arr = self._validate_input(X)
            X_seq = self._build_sequences(X_arr)
            X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

            self.model.eval()
            with torch.no_grad():
                dirs, rets, _ = self.model(X_t)

            result: dict[int, dict[str, np.ndarray]] = {}
            for h in self.params["horizons"]:
                key = str(h)
                result[h] = {
                    "direction": dirs[key].cpu().numpy().squeeze(-1),
                    "returns": rets[key].cpu().numpy().squeeze(-1),
                }
            return result

        except (ModelPredictionError, ModelError):
            raise
        except Exception as exc:
            raise ModelPredictionError(
                f"Prediction failed: {exc}",
                details={"error": str(exc)},
            ) from exc

    def predict_proba(self, X: Any) -> dict[int, np.ndarray]:
        """Return direction probabilities per horizon.

        Returns:
            Dictionary mapping horizon ``int`` to a 1-D probability array.
        """
        preds = self.predict(X)
        return {h: v["direction"] for h, v in preds.items()}

    def save(self, path: Path) -> None:
        """Persist model checkpoint to disk."""
        self._check_fitted()
        assert self.model is not None

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.model.state_dict(),
            "params": self.params,
            "num_features": self._num_features,
            "feature_names": self._feature_names,
            "vsn_weights": self._vsn_weights,
            "best_val_loss": self._best_val_loss,
        }
        torch.save(checkpoint, path)
        logger.info("model.saved", path=str(path))

    def load(self, path: Path) -> None:
        """Load model from a checkpoint file."""
        path = Path(path)
        if not path.exists():
            raise ModelLoadError(f"Model file not found: {path}", details={"path": str(path)})

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.params = checkpoint["params"]
            self._num_features = checkpoint["num_features"]
            self._feature_names = checkpoint.get("feature_names")
            self._vsn_weights = checkpoint.get("vsn_weights")
            self._best_val_loss = checkpoint.get("best_val_loss", float("inf"))

            self.model = self._create_model(self._num_features)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            self._fitted = True

            logger.info("model.loaded", path=str(path), num_features=self._num_features)
        except (ModelLoadError, ModelError):
            raise
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model: {exc}",
                details={"path": str(path), "error": str(exc)},
            ) from exc

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance based on variable selection network weights.

        The VSN learns soft attention weights over input features. Higher
        weight indicates the model relies more on that feature.
        """
        self._check_fitted()

        if self._vsn_weights is None:
            return {}

        weights = self._vsn_weights.copy()
        total = float(weights.sum())
        if total > 0:
            weights = weights / total

        n_features = len(weights)
        if self._feature_names and len(self._feature_names) == n_features:
            names = self._feature_names
        else:
            names = [f"feature_{i}" for i in range(n_features)]

        return {name: float(val) for name, val in zip(names, weights)}

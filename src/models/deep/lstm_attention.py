"""BiLSTM with Self-Attention model for multi-task financial prediction.

Predicts direction (up/down), expected returns, and volatility simultaneously
using a shared BiLSTM encoder with scaled dot-product self-attention and
separate output heads.
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
from torch.utils.data import DataLoader, Dataset, TensorDataset

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
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "sequence_length": 60,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "max_epochs": 100,
    "patience": 10,
    "validation_split": 0.1,
    "direction_loss_weight": 1.0,
    "return_loss_weight": 1.0,
    "volatility_loss_weight": 1.0,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_device() -> torch.device:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_numpy(x: Any) -> np.ndarray:
    """Convert input (DataFrame, Series, ndarray) to a numpy array."""
    if hasattr(x, "values"):
        return np.asarray(x.values, dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# PyTorch Module
# ---------------------------------------------------------------------------


class _SelfAttention(nn.Module):
    """Scaled dot-product self-attention over a sequence of hidden states."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            context: (batch, hidden_dim) — attention-weighted representation.
            weights: (batch, seq_len) — attention weights for interpretability.
        """
        q = self.query(x)  # (B, S, H)
        k = self.key(x)    # (B, S, H)
        v = self.value(x)  # (B, S, H)

        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B, S, S)
        weights = F.softmax(scores, dim=-1)                     # (B, S, S)
        attended = torch.bmm(weights, v)                        # (B, S, H)

        # Aggregate: mean over sequence dimension
        context = attended.mean(dim=1)  # (B, H)
        # Return the average attention weights across query positions for interpretability
        avg_weights = weights.mean(dim=1)  # (B, S)
        return context, avg_weights


class LSTMAttentionNetwork(nn.Module):
    """BiLSTM + Self-Attention with three prediction heads."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size

        # Shared encoder
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.attention = _SelfAttention(hidden_size * 2)
        self.encoder_dropout = nn.Dropout(dropout)

        # Output heads
        head_input_dim = hidden_size * 2
        self.direction_head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.return_head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.volatility_head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, num_features)

        Returns:
            direction: (batch, 1) probability of up-move.
            returns: (batch, 1) predicted return.
            volatility: (batch, 1) predicted volatility (positive).
            attn_weights: (batch, seq_len) attention weights.
        """
        lstm_out, _ = self.lstm(x)                      # (B, S, 2*H)
        lstm_out = self.layer_norm(lstm_out)
        context, attn_weights = self.attention(lstm_out) # (B, 2*H), (B, S)
        context = self.encoder_dropout(context)

        direction = self.direction_head(context)
        returns = self.return_head(context)
        volatility = self.volatility_head(context)

        return direction, returns, volatility, attn_weights


# ---------------------------------------------------------------------------
# Predictor (public interface)
# ---------------------------------------------------------------------------


class LSTMAttentionPredictor:
    """BiLSTM + Self-Attention predictor implementing the BasePredictor protocol.

    Multi-task model that simultaneously predicts:
    - direction: probability that the next move is up
    - returns: expected return magnitude
    - volatility: expected volatility
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = {**_DEFAULT_PARAMS, **(params or {})}
        self.device = _get_device()
        self.model: LSTMAttentionNetwork | None = None
        self._fitted = False
        self._num_features: int | None = None
        self._feature_names: list[str] | None = None
        self._feature_importance: dict[str, float] | None = None
        self._best_val_loss: float = float("inf")

        logger.info(
            "lstm_attention_predictor.init",
            device=str(self.device),
            params={k: v for k, v in self.params.items()},
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
        """Validate and convert input data to numpy array."""
        arr = _to_numpy(X)
        if arr.ndim == 1:
            raise ModelError(
                "Input must be at least 2-dimensional.",
                details={"shape": arr.shape},
            )
        if not fitting and self._num_features is not None:
            expected = self._num_features
            actual = arr.shape[-1]
            if actual != expected:
                raise ModelError(
                    f"Feature count mismatch: expected {expected}, got {actual}.",
                    details={"expected": expected, "actual": actual},
                )
        return arr

    def _build_sequences(self, X: np.ndarray) -> np.ndarray:
        """Reshape flat (samples, features) array into (n_sequences, seq_len, features)."""
        seq_len = self.params["sequence_length"]
        if X.ndim == 3:
            return X
        n_samples, n_features = X.shape
        if n_samples < seq_len:
            raise ModelError(
                f"Not enough samples ({n_samples}) for sequence length {seq_len}.",
                details={"n_samples": n_samples, "sequence_length": seq_len},
            )
        n_sequences = n_samples - seq_len + 1
        sequences = np.lib.stride_tricks.sliding_window_view(X, seq_len, axis=0)
        # sliding_window_view gives (n_sequences, n_features, seq_len) — transpose last two
        sequences = sequences.transpose(0, 2, 1)
        return sequences.astype(np.float32)

    def _create_model(self, num_features: int) -> LSTMAttentionNetwork:
        model = LSTMAttentionNetwork(
            num_features=num_features,
            hidden_size=self.params["hidden_size"],
            num_layers=self.params["num_layers"],
            dropout=self.params["dropout"],
        )
        return model.to(self.device)

    # ------------------------------------------------------------------
    # Public API (BasePredictor protocol)
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any, **kwargs: Any) -> None:
        """Train the model.

        Args:
            X: Feature matrix — (n_samples, n_features) or (n_samples, seq_len, n_features).
            y: Target DataFrame or array with columns [direction, returns, volatility].
            **kwargs: Optional overrides; ``validation_data=(X_val, y_val)``.
        """
        log = logger.bind(phase="fit")
        log.info("training.start")

        try:
            # --- Extract feature names --------------------------------
            if hasattr(X, "columns"):
                self._feature_names = list(X.columns)

            X_arr = self._validate_input(X, fitting=True)
            y_arr = _to_numpy(y)

            # Build sequences
            X_seq = self._build_sequences(X_arr)
            seq_len = self.params["sequence_length"]

            # Align targets: drop the first (seq_len - 1) rows when X was 2-D
            if X_arr.ndim == 2:
                y_arr = y_arr[seq_len - 1:]

            if y_arr.shape[0] != X_seq.shape[0]:
                raise ModelTrainingError(
                    "X and y length mismatch after sequencing.",
                    details={"X_sequences": X_seq.shape[0], "y_rows": y_arr.shape[0]},
                )

            # Ensure y has 3 columns
            if y_arr.ndim == 1:
                raise ModelTrainingError(
                    "y must have columns [direction, returns, volatility].",
                )
            if y_arr.shape[1] < 3:
                raise ModelTrainingError(
                    f"y must have at least 3 columns, got {y_arr.shape[1]}.",
                )

            y_dir = y_arr[:, 0]
            y_ret = y_arr[:, 1]
            y_vol = y_arr[:, 2]

            self._num_features = X_seq.shape[2]
            num_features = self._num_features

            # --- Validation split / external validation data ----------
            val_data = kwargs.get("validation_data")
            val_split = self.params["validation_split"]

            if val_data is not None:
                X_val_seq = self._build_sequences(self._validate_input(val_data[0], fitting=True))
                y_val_arr = _to_numpy(val_data[1])
                if val_data[0].ndim if hasattr(val_data[0], "ndim") else np.ndim(val_data[0]) == 2:
                    y_val_arr = y_val_arr[seq_len - 1:]
            else:
                n = X_seq.shape[0]
                split_idx = int(n * (1 - val_split))
                X_val_seq = X_seq[split_idx:]
                y_val_dir = y_dir[split_idx:]
                y_val_ret = y_ret[split_idx:]
                y_val_vol = y_vol[split_idx:]

                X_seq = X_seq[:split_idx]
                y_dir = y_dir[:split_idx]
                y_ret = y_ret[:split_idx]
                y_vol = y_vol[:split_idx]

                y_val_arr = np.column_stack([y_val_dir, y_val_ret, y_val_vol])

            # Tensors
            X_train_t = torch.tensor(X_seq, dtype=torch.float32)
            y_dir_t = torch.tensor(y_dir, dtype=torch.float32).unsqueeze(1)
            y_ret_t = torch.tensor(y_ret, dtype=torch.float32).unsqueeze(1)
            y_vol_t = torch.tensor(y_vol, dtype=torch.float32).unsqueeze(1)

            X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
            if val_data is not None:
                y_val_dir_t = torch.tensor(y_val_arr[:, 0], dtype=torch.float32).unsqueeze(1)
                y_val_ret_t = torch.tensor(y_val_arr[:, 1], dtype=torch.float32).unsqueeze(1)
                y_val_vol_t = torch.tensor(y_val_arr[:, 2], dtype=torch.float32).unsqueeze(1)
            else:
                y_val_dir_t = torch.tensor(y_val_dir, dtype=torch.float32).unsqueeze(1)
                y_val_ret_t = torch.tensor(y_val_ret, dtype=torch.float32).unsqueeze(1)
                y_val_vol_t = torch.tensor(y_val_vol, dtype=torch.float32).unsqueeze(1)

            train_ds = TensorDataset(X_train_t, y_dir_t, y_ret_t, y_vol_t)
            val_ds = TensorDataset(X_val_t, y_val_dir_t, y_val_ret_t, y_val_vol_t)

            batch_size = self.params["batch_size"]
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            # --- Model, optimizer, scheduler --------------------------
            self.model = self._create_model(num_features)
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.params["learning_rate"],
                weight_decay=self.params["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
            )

            bce_loss_fn = nn.BCELoss()
            mse_loss_fn = nn.MSELoss()

            w_dir = self.params["direction_loss_weight"]
            w_ret = self.params["return_loss_weight"]
            w_vol = self.params["volatility_loss_weight"]

            best_val_loss = float("inf")
            best_state: dict[str, Any] | None = None
            epochs_no_improve = 0
            patience = self.params["patience"]
            max_epochs = self.params["max_epochs"]

            # --- Training loop ----------------------------------------
            for epoch in range(1, max_epochs + 1):
                self.model.train()
                train_loss_accum = 0.0
                n_batches = 0

                for batch_x, batch_dir, batch_ret, batch_vol in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_dir = batch_dir.to(self.device)
                    batch_ret = batch_ret.to(self.device)
                    batch_vol = batch_vol.to(self.device)

                    pred_dir, pred_ret, pred_vol, _ = self.model(batch_x)

                    loss = (
                        w_dir * bce_loss_fn(pred_dir, batch_dir)
                        + w_ret * mse_loss_fn(pred_ret, batch_ret)
                        + w_vol * mse_loss_fn(pred_vol, batch_vol)
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss_accum += loss.item()
                    n_batches += 1

                avg_train_loss = train_loss_accum / max(n_batches, 1)

                # --- Validation ---------------------------------------
                self.model.eval()
                val_loss_accum = 0.0
                n_val_batches = 0
                with torch.no_grad():
                    for batch_x, batch_dir, batch_ret, batch_vol in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_dir = batch_dir.to(self.device)
                        batch_ret = batch_ret.to(self.device)
                        batch_vol = batch_vol.to(self.device)

                        pred_dir, pred_ret, pred_vol, _ = self.model(batch_x)
                        v_loss = (
                            w_dir * bce_loss_fn(pred_dir, batch_dir)
                            + w_ret * mse_loss_fn(pred_ret, batch_ret)
                            + w_vol * mse_loss_fn(pred_vol, batch_vol)
                        )
                        val_loss_accum += v_loss.item()
                        n_val_batches += 1

                avg_val_loss = val_loss_accum / max(n_val_batches, 1)
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
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    log.info(
                        "training.early_stop",
                        epoch=epoch,
                        best_val_loss=round(best_val_loss, 6),
                    )
                    break

            # Restore best weights
            if best_state is not None:
                self.model.load_state_dict(best_state)

            self._best_val_loss = best_val_loss
            self._fitted = True

            # Compute feature importance (gradient-based)
            self._compute_feature_importance(X_train_t)

            log.info(
                "training.complete",
                best_val_loss=round(best_val_loss, 6),
                num_features=num_features,
            )

        except (ModelTrainingError, ModelError):
            raise
        except Exception as exc:
            raise ModelTrainingError(
                f"Training failed: {exc}",
                details={"error": str(exc)},
            ) from exc

    def predict(self, X: Any) -> dict[str, np.ndarray]:
        """Generate predictions.

        Args:
            X: Feature matrix.

        Returns:
            Dictionary with keys ``direction``, ``returns``, ``volatility``.
        """
        self._check_fitted()
        assert self.model is not None

        try:
            X_arr = self._validate_input(X)
            X_seq = self._build_sequences(X_arr)
            X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

            self.model.eval()
            with torch.no_grad():
                pred_dir, pred_ret, pred_vol, _ = self.model(X_t)

            return {
                "direction": pred_dir.cpu().numpy().squeeze(-1),
                "returns": pred_ret.cpu().numpy().squeeze(-1),
                "volatility": pred_vol.cpu().numpy().squeeze(-1),
            }
        except (ModelPredictionError, ModelError):
            raise
        except Exception as exc:
            raise ModelPredictionError(
                f"Prediction failed: {exc}",
                details={"error": str(exc)},
            ) from exc

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return direction probabilities.

        Args:
            X: Feature matrix.

        Returns:
            1-D array of up-move probabilities.
        """
        preds = self.predict(X)
        return preds["direction"]

    def save(self, path: Path) -> None:
        """Persist model to disk.

        Saves both the network state dict and a metadata dict containing
        the hyper-parameters and feature information needed to reload the
        model.
        """
        self._check_fitted()
        assert self.model is not None

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.model.state_dict(),
            "params": self.params,
            "num_features": self._num_features,
            "feature_names": self._feature_names,
            "feature_importance": self._feature_importance,
            "best_val_loss": self._best_val_loss,
        }
        torch.save(checkpoint, path)
        logger.info("model.saved", path=str(path))

    def load(self, path: Path) -> None:
        """Load model from disk."""
        path = Path(path)
        if not path.exists():
            raise ModelLoadError(
                f"Model file not found: {path}",
                details={"path": str(path)},
            )

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.params = checkpoint["params"]
            self._num_features = checkpoint["num_features"]
            self._feature_names = checkpoint.get("feature_names")
            self._feature_importance = checkpoint.get("feature_importance")
            self._best_val_loss = checkpoint.get("best_val_loss", float("inf"))

            self.model = self._create_model(self._num_features)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            self._fitted = True

            logger.info(
                "model.loaded",
                path=str(path),
                num_features=self._num_features,
            )
        except (ModelLoadError, ModelError):
            raise
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model: {exc}",
                details={"path": str(path), "error": str(exc)},
            ) from exc

    def get_feature_importance(self) -> dict[str, float]:
        """Return gradient-based feature importance.

        Importance is computed as the mean absolute gradient of the total
        loss with respect to each input feature, averaged over a sample
        of the training data.
        """
        self._check_fitted()
        if self._feature_importance is not None:
            return self._feature_importance
        return {}

    # ------------------------------------------------------------------
    # Internal: gradient-based feature importance
    # ------------------------------------------------------------------

    def _compute_feature_importance(self, X_train: torch.Tensor) -> None:
        """Compute mean |gradient| w.r.t. input features over a data sample."""
        assert self.model is not None

        self.model.eval()
        # Use a subsample to keep it fast
        max_samples = min(512, X_train.shape[0])
        sample = X_train[:max_samples].clone().to(self.device)
        sample.requires_grad_(True)

        pred_dir, pred_ret, pred_vol, _ = self.model(sample)
        combined = pred_dir.sum() + pred_ret.sum() + pred_vol.sum()
        combined.backward()

        assert sample.grad is not None
        # Mean absolute gradient across batch and sequence dims -> per feature
        importance = sample.grad.abs().mean(dim=(0, 1)).cpu().numpy()

        n_features = importance.shape[0]
        if self._feature_names and len(self._feature_names) == n_features:
            names = self._feature_names
        else:
            names = [f"feature_{i}" for i in range(n_features)]

        # Normalise to sum to 1
        total = float(importance.sum())
        if total > 0:
            importance = importance / total

        self._feature_importance = {
            name: float(val) for name, val in zip(names, importance)
        }

        logger.debug(
            "feature_importance.computed",
            top_5=dict(
                sorted(self._feature_importance.items(), key=lambda kv: -kv[1])[:5]
            ),
        )

"""Residual Decoder - Discriminative approach for attribute inference (Level 2).

Instead of predicting delta from x_s (generative), directly predict x_s from delta (discriminative).
This naturally handles x_u variance because the decoder sees many (delta, x_s) pairs
with different x_u during training and learns to extract x_s despite the noise.

Attack is just one forward pass - no optimization needed.

Key Insight:
    The embedding residual Delta = E_full - E_base encodes information about both
    sensitive features (x_s) and unrelated features (x_u). A discriminative decoder
    can learn to extract x_s from Delta by seeing many (Delta, x_s) pairs during
    training with varying x_u values, effectively learning to marginalize over x_u.

Architecture:
    Input: Delta = E_victim - E_base (192-dim) [+ E_base optionally]
    Encoder: MLP [512, 512, 256, 128] with LayerNorm, GELU, Dropout
    Output: Separate heads per sensitive feature
        - Categorical/binary: Linear -> n_categories logits (CrossEntropy)
        - Continuous/integer: Linear -> 1 value (MSE)

Target Normalization (Issue #39):
    For regression tasks, continuous targets can have vastly different scales.
    This causes training instability and poor performance. The solution is to
    normalize continuous targets to [-1, 1] during training and denormalize
    predictions back to original scale.

    Enable with: normalize_targets=True and feature_metadata parameter.

Experimental Results (Adult dataset, 1000 victims):
    Easy setting (3 categorical sensitive features):
        - marital-status: 97.0%
        - occupation: 97.3%
        - relationship: 96.5%
        - Mean categorical accuracy: 96.93%

    Hard setting (age moved to sensitive, 4 features):
        - age: MAE=1.00, R²=0.984
        - marital-status: 92.8%
        - occupation: 30.7%
        - relationship: 85.3%
        - Mean categorical accuracy: 69.6%

Note: The drop in occupation accuracy when age becomes unknown demonstrates
feature entanglement in the embedding space. This is expected for Level 2 attacks
where features are predicted independently. Level 3 attacks (sequential inference)
could potentially leverage predicted features to improve subsequent predictions.

Usage:
    decoder = ResidualDecoder(
        n_sensitive=3,
        sensitive_indices=[4, 5, 6],
        known_indices=[0, 2, 7, 8, 12],
        unrelated_indices=[1, 3, 9, 10, 11],
        n_categories_per_dim=[7, 15, 6],
        feature_metadata=FEATURE_METADATA,  # For target normalization
        normalize_targets=True,  # Enable target normalization
    )
    decoder.fit(X_shadow, tabpfn_model, epochs=200)
    X_s_pred, stats = decoder.attack(E_victim, E_base_victim)
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cascade_decoding.common.constants import EMBEDDING_DIM
from cascade_decoding.common.target_normalizer import TargetNormalizer, NormStrategy

if TYPE_CHECKING:
    from tabpfn import TabPFNClassifier, TabPFNRegressor


class ResidualDecoderNetwork(nn.Module):
    """MLP decoder that predicts x_s from embedding residual delta.

    Architecture:
        [delta (192); E_base (192)] -> MLP -> x_s (n_sensitive)

    For categorical features, outputs logits for each category.
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        n_sensitive: int = 1,
        n_categories_per_dim: list[int] | None = None,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.1,
        use_e_base: bool = True,
    ):
        super().__init__()

        if hidden_layers is None:
            # Larger default architecture
            hidden_layers = [512, 512, 256, 128]

        self.embedding_dim = embedding_dim
        self.n_sensitive = n_sensitive
        self.n_categories_per_dim = n_categories_per_dim or []
        self.use_e_base = use_e_base

        # Build encoder layers
        # Input: delta (+ E_base if use_e_base)
        in_dim = embedding_dim * 2 if use_e_base else embedding_dim
        layers = []

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output heads - one per sensitive feature
        # For categorical: output logits for each category
        # For continuous/integer: output single value
        self.output_heads = nn.ModuleList()
        for i in range(n_sensitive):
            if i < len(self.n_categories_per_dim) and self.n_categories_per_dim[i] is not None:
                n_cats = self.n_categories_per_dim[i]
                self.output_heads.append(nn.Linear(in_dim, n_cats))
            else:
                self.output_heads.append(nn.Linear(in_dim, 1))

    def forward(self, delta: torch.Tensor, e_base: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            delta: Embedding residuals (batch, embedding_dim)
            e_base: Base embeddings (batch, embedding_dim), required if use_e_base=True

        Returns:
            List of outputs, one per sensitive feature.
            For categorical: (batch, n_categories) logits
            For continuous: (batch, 1) values
        """
        if self.use_e_base:
            if e_base is None:
                raise ValueError("e_base required when use_e_base=True")
            x = torch.cat([delta, e_base], dim=-1)
        else:
            x = delta
        h = self.encoder(x)
        outputs = [head(h) for head in self.output_heads]
        return outputs

    def predict(self, delta: torch.Tensor, e_base: torch.Tensor | None = None) -> torch.Tensor:
        """Predict x_s values (argmax for categorical).

        Args:
            delta: Embedding residuals (batch, embedding_dim)
            e_base: Base embeddings (batch, embedding_dim)

        Returns:
            Predicted x_s (batch, n_sensitive)
        """
        outputs = self.forward(delta, e_base)
        predictions = []

        for i, out in enumerate(outputs):
            if i < len(self.n_categories_per_dim) and self.n_categories_per_dim[i] is not None:
                # Categorical: argmax
                pred = out.argmax(dim=-1).float()
            else:
                # Continuous/integer: squeeze
                pred = out.squeeze(-1)
            predictions.append(pred)

        return torch.stack(predictions, dim=-1)


class ResidualDecoder:
    """Residual Decoder for discriminative attribute inference.

    Training:
    1. For each shadow sample, compute delta = E_full - E_base
    2. Train decoder to predict x_s from delta using cross-entropy (categorical) or MSE (continuous)

    Attack:
    1. Compute delta_obs = E_victim - E_base_victim
    2. Run decoder: x_s_pred = Decoder(delta_obs)

    Target Normalization (Issue #39):
        For regression tasks, enable normalize_targets=True and provide feature_metadata.
        This normalizes continuous targets to [-1, 1] during training and denormalizes
        predictions back to original scale.
    """

    def __init__(
        self,
        n_sensitive: int,
        sensitive_indices: list[int],
        known_indices: list[int],
        unrelated_indices: list[int],
        n_categories_per_dim: list[int] | None = None,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.1,
        device: str = "cuda",
        use_e_base: bool = True,
        feature_metadata: dict[int, dict] | None = None,
        normalize_targets: bool = False,
        norm_strategy: NormStrategy = "minmax_sym",
    ):
        self.n_sensitive = n_sensitive
        self.sensitive_indices = sorted(sensitive_indices)
        self.known_indices = sorted(known_indices)
        self.unrelated_indices = sorted(unrelated_indices)
        self.n_categories_per_dim = n_categories_per_dim or []
        self.embedding_dim = embedding_dim
        self.device = device
        self.use_e_base = use_e_base
        self.feature_metadata = feature_metadata or {}
        self.normalize_targets = normalize_targets
        self.norm_strategy = norm_strategy

        self.network = ResidualDecoderNetwork(
            embedding_dim=embedding_dim,
            n_sensitive=n_sensitive,
            n_categories_per_dim=n_categories_per_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            use_e_base=use_e_base,
        ).to(device)

        self.is_fitted = False
        self.history = None

        # Normalization stats for embeddings
        self.delta_mean = None
        self.delta_std = None
        self.e_base_mean = None
        self.e_base_std = None

        # Target normalizer for regression (Issue #39)
        self.target_normalizer: TargetNormalizer | None = None
        if normalize_targets:
            if feature_metadata:
                self.target_normalizer = TargetNormalizer(
                    feature_metadata=feature_metadata,
                    strategy=norm_strategy,
                )
            else:
                warnings.warn(
                    "normalize_targets=True but feature_metadata not provided. "
                    "Target normalization disabled.",
                    UserWarning,
                    stacklevel=2,
                )

    def _create_masked_input(self, X: np.ndarray) -> np.ndarray:
        """Create input with sensitive and unrelated features masked."""
        X_masked = X.copy()
        X_masked[:, self.sensitive_indices] = np.nan
        X_masked[:, self.unrelated_indices] = np.nan
        return X_masked

    def _get_embeddings(self, model, X: np.ndarray) -> np.ndarray:
        """Get TabPFN embeddings."""
        from tabpfn.utils import get_embeddings
        embeddings = get_embeddings(model, X, data_source="test")
        if embeddings.ndim == 3:
            embeddings = embeddings.mean(axis=0)
        return embeddings

    def fit(
        self,
        X_shadow: np.ndarray,
        tabpfn_model: TabPFNClassifier | TabPFNRegressor,
        epochs: int = 200,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        log_interval: int = 20,
        random_state: int = 42,
    ) -> "ResidualDecoder":
        """Train the decoder on shadow data.

        For each sample:
        1. Compute E_base = TabPFN(x_c, NaN, NaN)
        2. Compute E_full = TabPFN(x_c, x_s, x_u)
        3. delta = E_full - E_base
        4. Train decoder to predict x_s from delta
        """
        print("=" * 70)
        print("Training Residual Decoder")
        print("=" * 70)
        print(f"Shadow samples: {len(X_shadow)}")
        print(f"Sensitive features: {self.sensitive_indices}")
        print(f"Categories per dim: {self.n_categories_per_dim}")
        print(f"Use E_base as input: {self.use_e_base}")
        print(f"Target normalization: {self.normalize_targets} ({self.norm_strategy})")

        rng = np.random.default_rng(random_state)
        n_samples = len(X_shadow)
        device = torch.device(self.device)

        # Step 1: Compute base embeddings
        print("\nStep 1: Computing base embeddings E_base...")
        X_base = self._create_masked_input(X_shadow)
        E_base = self._get_embeddings(tabpfn_model, X_base)
        print(f"  E_base shape: {E_base.shape}")

        # Step 2: Compute full embeddings
        print("\nStep 2: Computing full embeddings E_full...")
        E_full = self._get_embeddings(tabpfn_model, X_shadow)
        print(f"  E_full shape: {E_full.shape}")

        # Step 3: Compute residuals
        print("\nStep 3: Computing residuals delta...")
        delta = E_full - E_base
        print(f"  delta shape: {delta.shape}")

        # Extract sensitive features as targets
        X_s = X_shadow[:, self.sensitive_indices]

        # Fit target normalizer if enabled (Issue #39)
        if self.target_normalizer is not None:
            print("\nStep 3.5: Fitting target normalizer for regression...")
            self.target_normalizer.fit(X_shadow, self.sensitive_indices)
            for i, idx in enumerate(self.sensitive_indices):
                info = self.target_normalizer.get_info(idx)
                if not info.get("skip", True):
                    feature_name = self.feature_metadata.get(idx, {}).get("name", f"feature_{idx}")
                    print(f"  {feature_name}: range=[{info['min']:.2f}, {info['max']:.2f}]")

        # Convert to tensors
        delta_t = torch.tensor(delta, dtype=torch.float32, device=device)
        E_base_t = torch.tensor(E_base, dtype=torch.float32, device=device)
        # Use float for all, will cast to long for CE loss per-feature
        X_s_t = torch.tensor(X_s, dtype=torch.float32, device=device)

        # Normalize targets for regression features (Issue #39)
        if self.target_normalizer is not None:
            X_s_t = self.target_normalizer.normalize_batch(X_s_t, self.sensitive_indices)

        # Normalize delta and E_base
        self.delta_mean = delta_t.mean(dim=0)
        self.delta_std = delta_t.std(dim=0).clamp(min=1e-6)
        delta_t = (delta_t - self.delta_mean) / self.delta_std

        self.e_base_mean = E_base_t.mean(dim=0)
        self.e_base_std = E_base_t.std(dim=0).clamp(min=1e-6)
        E_base_t = (E_base_t - self.e_base_mean) / self.e_base_std

        # Step 4: Training
        print(f"\nStep 4: Training ({epochs} epochs)...")

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.history = {"train_loss": [], "train_acc": []}

        for epoch in range(epochs):
            self.network.train()
            epoch_loss = 0.0

            indices = rng.permutation(n_samples)

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]

                if len(batch_idx) < 2:
                    continue

                delta_batch = delta_t[batch_idx]
                E_base_batch = E_base_t[batch_idx] if self.use_e_base else None
                X_s_batch = X_s_t[batch_idx]

                optimizer.zero_grad()

                outputs = self.network(delta_batch, E_base_batch)

                # Compute loss for each sensitive feature
                total_loss = 0.0
                for i, out in enumerate(outputs):
                    target = X_s_batch[:, i]
                    if i < len(self.n_categories_per_dim) and self.n_categories_per_dim[i] is not None:
                        # Cross-entropy for categorical
                        loss = F.cross_entropy(out, target.long())
                    else:
                        # MSE for continuous/integer
                        loss = F.mse_loss(out.squeeze(-1), target)
                    total_loss += loss

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item() * len(batch_idx)

            scheduler.step()
            epoch_loss /= n_samples
            self.history["train_loss"].append(epoch_loss)

            # Compute accuracy (for categorical) or MAE (for continuous)
            self.network.eval()
            with torch.no_grad():
                E_base_eval = E_base_t if self.use_e_base else None
                pred = self.network.predict(delta_t, E_base_eval)
                accuracies = []
                for i in range(self.n_sensitive):
                    if i < len(self.n_categories_per_dim) and self.n_categories_per_dim[i] is not None:
                        acc = (pred[:, i] == X_s_t[:, i]).float().mean().item()
                        accuracies.append(acc)
                mean_acc = np.mean(accuracies) if accuracies else 0.0
                self.history["train_acc"].append(mean_acc)

            if (epoch + 1) % log_interval == 0:
                print(f"  Epoch {epoch+1:4d}/{epochs}: loss={epoch_loss:.4f}, acc={mean_acc:.4f}")

        self.is_fitted = True
        print(f"\nTraining complete. Final accuracy: {self.history['train_acc'][-1]:.4f}")

        return self

    def predict(self, delta: np.ndarray, e_base: np.ndarray | None = None) -> np.ndarray:
        """Predict x_s from residuals.

        Args:
            delta: Embedding residuals (n_samples, embedding_dim)
            e_base: Base embeddings (n_samples, embedding_dim), required if use_e_base=True

        Returns:
            Predicted x_s (n_samples, n_sensitive) in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before prediction")

        device = torch.device(self.device)
        delta_t = torch.tensor(delta, dtype=torch.float32, device=device)

        # Normalize delta
        if self.delta_mean is not None:
            delta_t = (delta_t - self.delta_mean) / self.delta_std

        # Handle E_base
        E_base_t = None
        if self.use_e_base:
            if e_base is None:
                raise ValueError("e_base required when use_e_base=True")
            E_base_t = torch.tensor(e_base, dtype=torch.float32, device=device)
            if self.e_base_mean is not None:
                E_base_t = (E_base_t - self.e_base_mean) / self.e_base_std

        self.network.eval()
        with torch.no_grad():
            pred = self.network.predict(delta_t, E_base_t)

        pred_np = pred.cpu().numpy()

        # Convert predictions back to original scale for regression features (Issue #39)
        if self.target_normalizer is not None:
            pred_np = self.target_normalizer.denormalize_batch(pred_np, self.sensitive_indices)

        return pred_np

    def attack(
        self,
        E_victim: np.ndarray,
        E_base_victim: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Run attack - just a forward pass!

        Args:
            E_victim: Victim full embeddings (n_victims, embedding_dim)
            E_base_victim: Victim base embeddings (n_victims, embedding_dim)

        Returns:
            Predicted x_s (n_victims, n_sensitive)
            Attack statistics
        """
        # Compute observed residual
        delta_obs = E_victim - E_base_victim

        # One forward pass (pass E_base if needed)
        e_base_input = E_base_victim if self.use_e_base else None
        X_s_pred = self.predict(delta_obs, e_base_input)

        stats = {
            "n_victims": len(E_victim),
            "method": "decoder",
            "use_e_base": self.use_e_base,
        }

        return X_s_pred, stats

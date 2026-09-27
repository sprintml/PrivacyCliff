"""Cascade Decoding Attack (Level 3).

This module implements the Cascade Decoding algorithm for multi-attribute reconstruction
using a cascading inference strategy. Instead of predicting all sensitive
features independently, Cascade:

1. Greedily selects the "easiest" feature to predict first
2. Uses that prediction to enrich context for subsequent predictions
3. Recomputes delta embeddings at each step as context grows

Key Insight:
    As we predict more features and add them to the known context, the base
    embedding E_base gets closer to E_full, making the delta smaller and
    containing less noise from remaining unknowns.

Two Training Variants:
    - B1 (Teacher Forcing): Use ground truth values to update context during training
    - B2 (Predicted): Use decoder predictions to update context during training

Algorithm:
    Phase 1 (Strategy Discovery):
        For each step t:
            1. Compute E_base_t with current context (known + predicted so far)
            2. Delta_t = E_full - E_base_t
            3. For each remaining sensitive feature, train candidate decoder
            4. Select feature with best validation accuracy
            5. Update context with ground truth (B1) or prediction (B2)

    Phase 2 (Cascading Attack):
        For each step t:
            1. Compute E_base_t with current context (known + predictions)
            2. Delta_t = E_victim - E_base_t
            3. Predict next feature using trained decoder
            4. Add prediction to context

Complexity:
    O(k*(k+1)/2) decoder trainings where k = number of sensitive features.
    This is because step t trains (k-t) candidate decoders to select the best one.

Usage:
    attack = CascadeAttack(
        sensitive_indices=[0, 4, 5, 6],
        known_indices=[2, 7, 8, 12],
        irrelevant_indices=[1, 3, 9, 10, 11],
        feature_metadata=ADULT_FEATURES,
        use_teacher_forcing=True,  # B1
    )
    attack.fit(X_shadow, tabpfn_model, epochs=100)
    predictions, stats = attack.attack(E_victim, E_base_initial, X_known_victim)
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


class CascadeDecoderNetwork(nn.Module):
    """Single-feature decoder network for one cascade step.

    Predicts a single sensitive feature from the embedding residual.

    Architecture:
        Input: [Delta (192) ; E_base (192)] = 384-dim
        Encoder: MLP with LayerNorm, GELU, Dropout
        Output: Feature prediction
            - Categorical: logits (n_categories)
            - Continuous: scalar value

    Args:
        embedding_dim: Dimension of embeddings (default: 192)
        n_categories: Number of categories for categorical features, None for continuous
        hidden_layers: List of hidden layer sizes
        dropout: Dropout rate
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        n_categories: int | None = None,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [512, 512, 256, 128]

        self.embedding_dim = embedding_dim
        self.n_categories = n_categories
        self.is_categorical = n_categories is not None

        # Input: [delta ; e_base]
        in_dim = embedding_dim * 2
        layers = []

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output head
        if self.is_categorical:
            self.output_head = nn.Linear(in_dim, n_categories)
        else:
            self.output_head = nn.Linear(in_dim, 1)

    def forward(self, delta: torch.Tensor, e_base: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            delta: Embedding residuals (batch, embedding_dim)
            e_base: Base embeddings (batch, embedding_dim)

        Returns:
            For categorical: logits (batch, n_categories)
            For continuous: values (batch, 1)
        """
        x = torch.cat([delta, e_base], dim=-1)
        h = self.encoder(x)
        return self.output_head(h)

    def predict(self, delta: torch.Tensor, e_base: torch.Tensor) -> torch.Tensor:
        """Predict feature values.

        Args:
            delta: Embedding residuals (batch, embedding_dim)
            e_base: Base embeddings (batch, embedding_dim)

        Returns:
            Predictions (batch,) - argmax for categorical, raw value for continuous
        """
        out = self.forward(delta, e_base)
        if self.is_categorical:
            return out.argmax(dim=-1).float()
        return out.squeeze(-1)


class CascadeAttack:
    """Cascade Decoding Attack.

    Implements cascading inference for multi-attribute reconstruction.

    Args:
        sensitive_indices: Feature indices to infer (attack targets)
        known_indices: Feature indices adversary knows
        irrelevant_indices: Feature indices to ignore (kept as NaN)
        feature_metadata: Dict mapping feature index to metadata dict with:
            - dtype: "categorical", "binary", "integer", or "continuous"
            - n_categories: Number of categories (for categorical/binary)
        use_teacher_forcing: If True (B1), use ground truth during training
                            If False (B2), use predictions during training
        embedding_dim: Dimension of TabPFN embeddings
        hidden_layers: Hidden layer sizes for decoders
        dropout: Dropout rate for decoders
        device: Device for computation
        normalize_targets: If True, normalize continuous targets (Issue #39)
        norm_strategy: Normalization strategy for targets
    """

    def __init__(
        self,
        sensitive_indices: list[int],
        known_indices: list[int],
        irrelevant_indices: list[int],
        feature_metadata: dict[int, dict],
        use_teacher_forcing: bool = True,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.1,
        device: str = "cuda",
        normalize_targets: bool = False,
        norm_strategy: NormStrategy = "minmax_sym",
    ):
        self.sensitive_indices = sorted(sensitive_indices)
        self.known_indices = sorted(known_indices)
        self.irrelevant_indices = sorted(irrelevant_indices)
        self.feature_metadata = feature_metadata
        self.use_teacher_forcing = use_teacher_forcing
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.device = device
        self.normalize_targets = normalize_targets
        self.norm_strategy = norm_strategy

        # Will be populated during fit()
        self.decoders: list[CascadeDecoderNetwork] = []
        self.attack_order: list[int] = []
        self.tabpfn_model = None  # Stored during fit() for convenience
        self.is_fitted = False

        # Normalization stats (computed during fit)
        self.delta_means: list[torch.Tensor] = []
        self.delta_stds: list[torch.Tensor] = []
        self.e_base_means: list[torch.Tensor] = []
        self.e_base_stds: list[torch.Tensor] = []

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

        # Training history
        self.history: dict = {}

    def _get_feature_n_categories(self, feature_idx: int) -> int | None:
        """Get number of categories for a feature, None if continuous."""
        if feature_idx not in self.feature_metadata:
            return None
        meta = self.feature_metadata[feature_idx]
        dtype = meta.get("dtype", "continuous")
        if dtype in ("categorical", "binary"):
            return meta.get("n_categories", 2)
        return None

    def _get_embeddings(
        self, model: TabPFNClassifier | TabPFNRegressor, X: np.ndarray
    ) -> np.ndarray:
        """Get TabPFN embeddings for input data."""
        from tabpfn.utils import get_embeddings

        embeddings = get_embeddings(model, X, data_source="test")
        if embeddings.ndim == 3:
            embeddings = embeddings.mean(axis=0)
        return embeddings

    def _create_masked_input(
        self, X: np.ndarray, context_indices: list[int]
    ) -> np.ndarray:
        """Create input with only context features visible, rest as NaN.

        Args:
            X: Full feature matrix (n_samples, n_features)
            context_indices: Indices of features to keep visible

        Returns:
            Masked input with NaN for non-context features
        """
        X_masked = np.full_like(X, np.nan)
        for idx in context_indices:
            X_masked[:, idx] = X[:, idx]
        return X_masked

    def _train_single_decoder(
        self,
        target_idx: int,
        delta_train: torch.Tensor,
        e_base_train: torch.Tensor,
        y_train: torch.Tensor,
        delta_held_out: torch.Tensor,
        e_base_held_out: torch.Tensor,
        y_held_out: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
    ) -> tuple[CascadeDecoderNetwork, float, dict]:
        """Train a decoder for a single feature.

        Args:
            target_idx: Feature index to predict
            delta_train/held_out: Training/held-out deltas
            e_base_train/held_out: Training/held-out base embeddings
            y_train/held_out: Training/held-out targets (in original scale)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience (stop if no improvement for this many epochs)

        Returns:
            Trained decoder, held-out score, training stats
        """
        n_categories = self._get_feature_n_categories(target_idx)
        is_categorical = n_categories is not None

        # Normalize targets for regression features (Issue #39)
        y_train_norm = y_train
        y_held_out_norm = y_held_out
        if not is_categorical and self.target_normalizer is not None:
            y_train_norm = self.target_normalizer.normalize(y_train, target_idx)
            y_held_out_norm = self.target_normalizer.normalize(y_held_out, target_idx)

        # Create decoder
        decoder = CascadeDecoderNetwork(
            embedding_dim=self.embedding_dim,
            n_categories=n_categories,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            decoder.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        n_train = len(delta_train)
        best_held_out_score = -float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            decoder.train()
            indices = torch.randperm(n_train, device=self.device)

            for start_idx in range(0, n_train, batch_size):
                end_idx = min(start_idx + batch_size, n_train)
                batch_idx = indices[start_idx:end_idx]

                if len(batch_idx) < 2:
                    continue

                delta_batch = delta_train[batch_idx]
                e_base_batch = e_base_train[batch_idx]
                y_batch = y_train_norm[batch_idx]  # Use normalized targets

                optimizer.zero_grad()
                out = decoder(delta_batch, e_base_batch)

                if is_categorical:
                    loss = F.cross_entropy(out, y_batch.long())
                else:
                    loss = F.mse_loss(out.squeeze(-1), y_batch)

                loss.backward()
                optimizer.step()

            scheduler.step()

            # Held-out scoring
            decoder.eval()
            with torch.no_grad():
                held_out_pred = decoder.predict(delta_held_out, e_base_held_out)

                if is_categorical:
                    held_out_score = (held_out_pred == y_held_out_norm).float().mean().item()
                else:
                    # Use negative MAE as score (higher is better)
                    # Compare in normalized space for consistent scoring
                    held_out_score = -torch.abs(held_out_pred - y_held_out_norm).mean().item()

                if held_out_score > best_held_out_score:
                    best_held_out_score = held_out_score
                    best_state = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                break

        # Load best state
        if best_state is not None:
            decoder.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        stats = {
            "epochs_run": epoch + 1,
            "epochs_max": epochs,
            "early_stopped": epochs_without_improvement >= patience,
            "best_held_out_score": best_held_out_score,
            "is_categorical": is_categorical,
        }

        return decoder, best_held_out_score, stats

    def fit(
        self,
        X_shadow: np.ndarray,
        tabpfn_model: TabPFNClassifier | TabPFNRegressor,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        val_ratio: float = 0.2,
        random_state: int = 42,
        verbose: bool = True,
    ) -> "CascadeAttack":
        """Train cascade decoders (Phase 1: Strategy Discovery).

        Uses greedy selection to find optimal attack order and trains
        a decoder for each step.

        Args:
            X_shadow: Shadow dataset (n_samples, n_features)
            tabpfn_model: Fitted TabPFN model
            epochs: Training epochs per decoder
            batch_size: Batch size for training
            learning_rate: Learning rate
            val_ratio: Fraction of shadow data for validation
            random_state: Random seed
            verbose: Whether to print progress

        Returns:
            self for method chaining
        """
        if verbose:
            print("=" * 70)
            print("Cascade: Phase 1 - Strategy Discovery")
            print("=" * 70)
            mode = "Teacher Forcing (B1)" if self.use_teacher_forcing else "Predicted (B2)"
            print(f"Mode: {mode}")
            print(f"Sensitive features: {self.sensitive_indices}")
            print(f"Known features: {self.known_indices}")
            print(f"Target normalization: {self.normalize_targets} ({self.norm_strategy})")

        # Store model for use in attack()
        self.tabpfn_model = tabpfn_model

        rng = np.random.default_rng(random_state)
        n_samples = len(X_shadow)
        device = torch.device(self.device)

        # Split into train/val
        n_val = int(n_samples * val_ratio)
        indices = rng.permutation(n_samples)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train = X_shadow[train_idx]
        X_held_out = X_shadow[val_idx]

        if verbose:
            print(f"\nShadow data: {n_samples} samples")
            print(f"  Train: {len(train_idx)}, Held-out: {len(val_idx)}")

        # Fit target normalizer if enabled (Issue #39)
        if self.target_normalizer is not None:
            if verbose:
                print("\nFitting target normalizer for regression features...")
            self.target_normalizer.fit(X_train, self.sensitive_indices)
            for idx in self.sensitive_indices:
                info = self.target_normalizer.get_info(idx)
                if not info.get("skip", True):
                    feature_name = self.feature_metadata.get(idx, {}).get("name", f"feature_{idx}")
                    if verbose:
                        print(f"  {feature_name}: range=[{info['min']:.2f}, {info['max']:.2f}]")

        # Compute full embeddings once
        if verbose:
            print("\nComputing full embeddings E_full...")
        E_full_train = self._get_embeddings(tabpfn_model, X_train)
        E_full_held_out = self._get_embeddings(tabpfn_model, X_held_out)

        E_full_train_t = torch.tensor(E_full_train, dtype=torch.float32, device=device)
        E_full_held_out_t = torch.tensor(E_full_held_out, dtype=torch.float32, device=device)

        # Initialize context with known features
        current_context = list(self.known_indices)
        remaining_sensitive = list(self.sensitive_indices)

        # For B2 mode, track predictions to update X
        if not self.use_teacher_forcing:
            X_train_with_pred = X_train.copy()
            X_held_out_with_pred = X_held_out.copy()

        self.decoders = []
        self.attack_order = []
        self.delta_means = []
        self.delta_stds = []
        self.e_base_means = []
        self.e_base_stds = []
        self.history = {"steps": []}

        step = 0
        while remaining_sensitive:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Step {step}: {len(remaining_sensitive)} features remaining")
                print(f"Current context: {current_context}")

            # Compute base embeddings with current context
            if self.use_teacher_forcing:
                X_train_masked = self._create_masked_input(X_train, current_context)
                X_held_out_masked = self._create_masked_input(X_held_out, current_context)
            else:
                X_train_masked = self._create_masked_input(X_train_with_pred, current_context)
                X_held_out_masked = self._create_masked_input(X_held_out_with_pred, current_context)

            E_base_train = self._get_embeddings(tabpfn_model, X_train_masked)
            E_base_held_out = self._get_embeddings(tabpfn_model, X_held_out_masked)

            E_base_train_t = torch.tensor(E_base_train, dtype=torch.float32, device=device)
            E_base_held_out_t = torch.tensor(E_base_held_out, dtype=torch.float32, device=device)

            # Compute deltas
            delta_train_t = E_full_train_t - E_base_train_t
            delta_held_out_t = E_full_held_out_t - E_base_held_out_t

            # Normalize delta and E_base
            delta_mean = delta_train_t.mean(dim=0)
            delta_std = delta_train_t.std(dim=0).clamp(min=1e-6)
            delta_train_norm = (delta_train_t - delta_mean) / delta_std
            delta_held_out_norm = (delta_held_out_t - delta_mean) / delta_std

            e_base_mean = E_base_train_t.mean(dim=0)
            e_base_std = E_base_train_t.std(dim=0).clamp(min=1e-6)
            e_base_train_norm = (E_base_train_t - e_base_mean) / e_base_std
            e_base_held_out_norm = (E_base_held_out_t - e_base_mean) / e_base_std

            # Try each remaining sensitive feature
            best_score = -float("inf")
            best_decoder = None
            best_feature = None
            best_stats = None

            for candidate_idx in remaining_sensitive:
                if verbose:
                    feature_name = self.feature_metadata.get(candidate_idx, {}).get(
                        "name", f"feature_{candidate_idx}"
                    )
                    print(f"  Training candidate decoder for {feature_name} (idx={candidate_idx})...")

                # Get target values
                y_train = torch.tensor(
                    X_train[:, candidate_idx], dtype=torch.float32, device=device
                )
                y_held_out = torch.tensor(
                    X_held_out[:, candidate_idx], dtype=torch.float32, device=device
                )

                # Train decoder
                decoder, held_out_score, stats = self._train_single_decoder(
                    target_idx=candidate_idx,
                    delta_train=delta_train_norm,
                    e_base_train=e_base_train_norm,
                    y_train=y_train,
                    delta_held_out=delta_held_out_norm,
                    e_base_held_out=e_base_held_out_norm,
                    y_held_out=y_held_out,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                )

                if verbose:
                    score_type = "accuracy" if stats["is_categorical"] else "neg_MAE"
                    print(f"    {score_type}: {held_out_score:.4f}")

                if held_out_score > best_score:
                    best_score = held_out_score
                    best_decoder = decoder
                    best_feature = candidate_idx
                    best_stats = stats

            # Select best feature for this step
            feature_name = self.feature_metadata.get(best_feature, {}).get(
                "name", f"feature_{best_feature}"
            )
            if verbose:
                print(f"\n  Selected: {feature_name} (idx={best_feature}), score={best_score:.4f}")

            # Store decoder and normalization stats
            self.decoders.append(best_decoder)
            self.attack_order.append(best_feature)
            self.delta_means.append(delta_mean)
            self.delta_stds.append(delta_std)
            self.e_base_means.append(e_base_mean)
            self.e_base_stds.append(e_base_std)

            # Update history
            self.history["steps"].append({
                "step": step,
                "feature_idx": best_feature,
                "feature_name": feature_name,
                "val_score": best_score,
                "is_categorical": best_stats["is_categorical"],
            })

            # Update context
            remaining_sensitive.remove(best_feature)
            current_context.append(best_feature)

            # For B2 mode, update X with predictions
            if not self.use_teacher_forcing and remaining_sensitive:
                best_decoder.eval()
                with torch.no_grad():
                    pred_train = best_decoder.predict(delta_train_norm, e_base_train_norm)
                    pred_held_out = best_decoder.predict(delta_held_out_norm, e_base_held_out_norm)

                pred_train_np = pred_train.cpu().numpy()
                pred_held_out_np = pred_held_out.cpu().numpy()

                # Denormalize predictions for regression features (Issue #39)
                # Predictions are in normalized space, need original scale for embeddings
                n_categories = self._get_feature_n_categories(best_feature)
                if n_categories is None and self.target_normalizer is not None:
                    pred_train_np = self.target_normalizer.denormalize(pred_train_np, best_feature)
                    pred_held_out_np = self.target_normalizer.denormalize(pred_held_out_np, best_feature)

                X_train_with_pred[:, best_feature] = pred_train_np
                X_held_out_with_pred[:, best_feature] = pred_held_out_np

            step += 1

        self.is_fitted = True

        if verbose:
            print("\n" + "=" * 70)
            print("Strategy Discovery Complete")
            print("=" * 70)
            print(f"Attack order: {self.attack_order}")
            for i, step_info in enumerate(self.history["steps"]):
                print(f"  Step {i}: {step_info['feature_name']} "
                      f"(score={step_info['val_score']:.4f})")

        return self

    def attack(
        self,
        E_victim: np.ndarray,
        X_known_victim: np.ndarray,
        tabpfn_model: TabPFNClassifier | TabPFNRegressor | None = None,
        verbose: bool = True,
    ) -> tuple[dict[int, np.ndarray], dict]:
        """Run cascading attack (Phase 2: Inference).

        Args:
            E_victim: Full victim embeddings (n_victims, embedding_dim)
            X_known_victim: Known feature values for victims (n_victims, n_features)
                           Should have NaN for unknown features
            tabpfn_model: Fitted TabPFN model (for computing base embeddings).
                         If None, uses the model stored during fit().
            verbose: Whether to print progress

        Returns:
            predictions: Dict mapping feature_idx to predictions (n_victims,)
            stats: Attack statistics
        """
        if not self.is_fitted:
            raise RuntimeError("CascadeAttack must be fitted before attack()")

        # Use stored model if not provided
        if tabpfn_model is None:
            tabpfn_model = self.tabpfn_model
        if tabpfn_model is None:
            raise RuntimeError("No TabPFN model provided or stored from fit()")

        if verbose:
            print("\n" + "=" * 70)
            print("Cascade: Phase 2 - Cascading Attack")
            print("=" * 70)

        device = torch.device(self.device)
        n_victims = len(E_victim)

        E_victim_t = torch.tensor(E_victim, dtype=torch.float32, device=device)

        # Initialize X with known features
        X_current = X_known_victim.copy()

        predictions = {}
        step_stats = []

        for step, (feature_idx, decoder) in enumerate(
            zip(self.attack_order, self.decoders)
        ):
            feature_name = self.feature_metadata.get(feature_idx, {}).get(
                "name", f"feature_{feature_idx}"
            )

            if verbose:
                print(f"\nStep {step}: Predicting {feature_name} (idx={feature_idx})")

            # Compute base embedding with current context
            X_masked = self._create_masked_input(
                X_current, self.known_indices + self.attack_order[:step]
            )
            E_base = self._get_embeddings(tabpfn_model, X_masked)
            E_base_t = torch.tensor(E_base, dtype=torch.float32, device=device)

            # Compute delta
            delta_t = E_victim_t - E_base_t

            # Normalize using stored stats
            delta_norm = (delta_t - self.delta_means[step]) / self.delta_stds[step]
            e_base_norm = (E_base_t - self.e_base_means[step]) / self.e_base_stds[step]

            # Predict
            decoder.eval()
            with torch.no_grad():
                pred = decoder.predict(delta_norm, e_base_norm)

            pred_np = pred.cpu().numpy()

            # Convert predictions back to original scale for regression features (Issue #39)
            n_categories = self._get_feature_n_categories(feature_idx)
            if n_categories is None and self.target_normalizer is not None:
                pred_np = self.target_normalizer.denormalize(pred_np, feature_idx)

            predictions[feature_idx] = pred_np

            # Update X_current with predictions for next step
            X_current[:, feature_idx] = pred_np

            step_stats.append({
                "step": step,
                "feature_idx": feature_idx,
                "feature_name": feature_name,
            })

            if verbose:
                if n_categories is not None:
                    unique, counts = np.unique(pred_np.astype(int), return_counts=True)
                    print(f"  Predictions: {dict(zip(unique, counts))}")
                else:
                    print(f"  Predictions: mean={pred_np.mean():.2f}, std={pred_np.std():.2f}")

        stats = {
            "n_victims": n_victims,
            "attack_order": self.attack_order,
            "use_teacher_forcing": self.use_teacher_forcing,
            "step_stats": step_stats,
        }

        if verbose:
            print("\n" + "=" * 70)
            print("Cascading Attack Complete")
            print("=" * 70)

        return predictions, stats


def run_cascade_attack(
    X_shadow: np.ndarray,
    X_victim: np.ndarray,
    tabpfn_model: TabPFNClassifier | TabPFNRegressor,
    sensitive_indices: list[int],
    known_indices: list[int],
    irrelevant_indices: list[int],
    feature_metadata: dict[int, dict],
    use_teacher_forcing: bool = True,
    epochs: int = 100,
    device: str = "cuda",
    verbose: bool = True,
    normalize_targets: bool = False,
    norm_strategy: NormStrategy = "minmax_sym",
) -> tuple[dict[int, np.ndarray], dict]:
    """Convenience function to run complete Cascade attack.

    Args:
        X_shadow: Shadow dataset for training
        X_victim: Victim dataset (full features, for computing embeddings)
        tabpfn_model: Fitted TabPFN model
        sensitive_indices: Features to infer
        known_indices: Features adversary knows
        irrelevant_indices: Features to ignore
        feature_metadata: Feature metadata dict
        use_teacher_forcing: B1 (True) or B2 (False)
        epochs: Training epochs per decoder
        device: Computation device
        verbose: Print progress
        normalize_targets: If True, normalize continuous targets (Issue #39)
        norm_strategy: Normalization strategy for targets

    Returns:
        predictions: Dict mapping feature_idx to predictions
        stats: Attack statistics including assessment metrics
    """
    from tabpfn.utils import get_embeddings

    from cascade_decoding.common.evaluation import evaluate_attribute_inference

    # Create attack
    attack = CascadeAttack(
        sensitive_indices=sensitive_indices,
        known_indices=known_indices,
        irrelevant_indices=irrelevant_indices,
        feature_metadata=feature_metadata,
        use_teacher_forcing=use_teacher_forcing,
        device=device,
        normalize_targets=normalize_targets,
        norm_strategy=norm_strategy,
    )

    # Fit on shadow data
    attack.fit(X_shadow, tabpfn_model, epochs=epochs, verbose=verbose)

    # Prepare victim data
    E_victim = get_embeddings(tabpfn_model, X_victim, data_source="test")
    if E_victim.ndim == 3:
        E_victim = E_victim.mean(axis=0)

    # Create X_known with only known features visible
    X_known_victim = np.full_like(X_victim, np.nan)
    for idx in known_indices:
        X_known_victim[:, idx] = X_victim[:, idx]

    # Run attack
    predictions, stats = attack.attack(
        E_victim, X_known_victim, tabpfn_model, verbose=verbose
    )

    # Assess results
    assessment_results = {}
    for feature_idx, pred in predictions.items():
        y_true = X_victim[:, feature_idx]
        meta = feature_metadata.get(feature_idx, {})
        dtype = meta.get("dtype", "continuous")
        n_categories = meta.get("n_categories")

        metrics = evaluate_attribute_inference(
            y_true=y_true,
            y_pred=pred,
            dtype=dtype,
            n_categories=n_categories,
            feature_name=meta.get("name", f"feature_{feature_idx}"),
            min_value=meta.get("min_value"),
            max_value=meta.get("max_value"),
        )
        assessment_results[feature_idx] = metrics

    stats["assessment"] = assessment_results

    return predictions, stats

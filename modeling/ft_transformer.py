"""FT-Transformer wrapper for binary supply-tightening classification.

Attention-based tabular model (tab_transformer_pytorch's FTTransformer). Falls
back to a sklearn MLPClassifier if the library or torch is unavailable, so the
benchmark still runs. Self-contained: no config.settings dependency.

Adapted from the teammate's build for the merged project.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
RANDOM_STATE = 42

_HAS_FT = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from tab_transformer_pytorch import FTTransformer
    _HAS_FT = True
except Exception:
    warnings.warn("tab_transformer_pytorch/torch unavailable — FT-Transformer falls back to MLP.")

_DEFAULTS = dict(dim=32, depth=4, heads=4, attn_dropout=0.1, ff_dropout=0.1,
                 epochs=30, learning_rate=1e-3, batch_size=64)


class _MLPFallback:
    def __init__(self, p):
        from sklearn.neural_network import MLPClassifier
        self.scaler = StandardScaler()
        self.model = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                   max_iter=p.get("epochs", 30),
                                   learning_rate_init=p.get("learning_rate", 1e-3),
                                   random_state=RANDOM_STATE, early_stopping=True,
                                   validation_fraction=0.1)

    def fit(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)

    def predict_proba1(self, X):
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]


class FTTransformerModel:
    def __init__(self, params=None):
        self.params = {**_DEFAULTS, **(params or {})}
        self.model = None
        self.scaler = None
        self._fallback = False
        # Per-epoch (train_loss, val_loss) — populated by fit(), read by
        # scripts/generate_model_curves.py to plot the training curve. Empty
        # list on the MLP fallback path (no epoch-level BCE loss exposed there).
        self.history = {"epoch": [], "train_loss": [], "val_loss": []}

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)

        if not _HAS_FT:
            self._fallback = True
            self.model = _MLPFallback(self.params)
            self.model.fit(X_train, y_train.astype(int))
            return self

        self._fallback = False
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_train).astype(np.float32)
        n_val = max(1, int(len(Xs) * 0.1))
        X_val, y_val = Xs[-n_val:], y_train[-n_val:]
        X_tr, y_tr = Xs[:-n_val], y_train[:-n_val]

        p = self.params
        device = torch.device("cpu")
        n_features = X_tr.shape[1]
        model = FTTransformer(categories=(), num_continuous=n_features,
                              dim=p["dim"], depth=p["depth"], heads=p["heads"],
                              dim_out=1, attn_dropout=p["attn_dropout"],
                              ff_dropout=p["ff_dropout"]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])

        t_Xtr = torch.tensor(X_tr, dtype=torch.float32)
        t_ytr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
        t_Xval = torch.tensor(X_val, dtype=torch.float32)
        t_yval = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(TensorDataset(t_Xtr, t_ytr), batch_size=p["batch_size"], shuffle=True)

        best_val, patience, bad, best_state = float("inf"), 5, 0, None
        for epoch in range(1, p["epochs"] + 1):
            model.train()
            epoch_losses = []
            for xb, yb in loader:
                logits = model(torch.empty(xb.size(0), 0, dtype=torch.long), xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_losses.append(loss.item())
            model.eval()
            with torch.no_grad():
                vloss = criterion(model(torch.empty(t_Xval.size(0), 0, dtype=torch.long), t_Xval), t_yval).item()
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(float(np.mean(epoch_losses)))
            self.history["val_loss"].append(float(vloss))
            if vloss < best_val:
                best_val, bad = vloss, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model = model
        return self

    def predict_proba1(self, X):
        """Probability of class 1."""
        if self._fallback:
            return self.model.predict_proba1(X)
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler.transform(X).astype(np.float32)
        t_X = torch.tensor(Xs, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.empty(t_X.size(0), 0, dtype=torch.long), t_X)
        return torch.sigmoid(logits).squeeze(-1).numpy()

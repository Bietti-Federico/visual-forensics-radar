"""A small PyTorch autoencoder anomaly detector.

Trained unsupervised (reconstruction loss only, no labels — satisfies the
platform brief's "independent from fraud labels") on whatever numeric
feature vectors it's given. Its input dimensionality varies with the
training batch's vocabulary (`DictVectorizer`'s fitted feature count), so the
network is built lazily inside `fit()` rather than at construction time.

`is_anomaly` compares a new vector's reconstruction error against
`mean + 3*std` of the *training batch's own* reconstruction errors — a
threshold derived from the same data the model was fit on, not an arbitrary
global constant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from sklearn.feature_extraction import DictVectorizer
from torch import nn

from pdf_forensics.domain.anomaly_detection.anomaly_score import AnomalyScore

_THRESHOLD_STD_MULTIPLIER = 3.0


class _AutoencoderNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded: torch.Tensor = self.encoder(x)
        decoded: torch.Tensor = self.decoder(encoded)
        return decoded


class AutoencoderAnomalyDetector:
    detector_id = "autoencoder"

    def __init__(
        self,
        hidden_dim: int = 8,
        bottleneck_dim: int = 4,
        epochs: int = 100,
        learning_rate: float = 0.01,
        random_seed: int = 42,
    ) -> None:
        self._hidden_dim = hidden_dim
        self._bottleneck_dim = bottleneck_dim
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._random_seed = random_seed
        self._vectorizer: DictVectorizer | None = None
        self._model: _AutoencoderNetwork | None = None
        self._threshold: float | None = None

    def fit(self, feature_vectors: Sequence[Mapping[str, float]]) -> None:
        torch.manual_seed(self._random_seed)

        self._vectorizer = DictVectorizer(sparse=False)
        matrix = self._vectorizer.fit_transform(list(feature_vectors))
        inputs = torch.tensor(matrix, dtype=torch.float32)

        model = _AutoencoderNetwork(inputs.shape[1], self._hidden_dim, self._bottleneck_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(self._epochs):
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), inputs)
            loss.backward()
            optimizer.step()
        self._model = model

        model.eval()
        with torch.no_grad():
            errors = torch.mean((model(inputs) - inputs) ** 2, dim=1)
        spread = _THRESHOLD_STD_MULTIPLIER * errors.std(unbiased=False)
        self._threshold = float(errors.mean() + spread)

    def score(self, feature_vector: Mapping[str, float]) -> AnomalyScore:
        if self._vectorizer is None or self._model is None or self._threshold is None:
            raise RuntimeError(f"{self.detector_id} has not been fit yet.")

        matrix = self._vectorizer.transform([dict(feature_vector)])
        inputs = torch.tensor(matrix, dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            error = float(torch.mean((self._model(inputs) - inputs) ** 2))

        return AnomalyScore(
            detector_id=self.detector_id, score=error, is_anomaly=error > self._threshold
        )

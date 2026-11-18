"""CNN architecture based on the layer description supplied for the 2025 IEEE ICTBIG paper.

Input: float tensor shaped (batch, 1, 400, 400), normalized to [0, 1].
Output: raw class logits shaped (batch, num_classes); use CrossEntropyLoss.
The paper's reported training setup is Adam (learning rate 0.001), batch size
32, categorical cross-entropy, and 50 epochs; those settings are documentation
only and no training code is included here.

The paper description supplied specifies three convolution layers, 2x2 max pooling,
dropout, and two dense hidden layers but does not give pool placement or padding.
This implementation documents the reproducible assumption of one 2x2 pool after
each convolution (valid 3x3 convolutions). The resulting 128x48x48 representation
feeds a 128-unit layer: this is 37,748,736 weights in the first dense layer alone
(about 151 MB in float32). This is a substantial parameter cost and should be
reviewed against the full paper before training. No softmax is applied: PyTorch's
CrossEntropyLoss consumes logits and includes log-softmax internally.
"""

from pathlib import Path

import torch
from torch import Tensor, nn


class ASLAlphabetCNN(nn.Module):
    """Static-sign CNN. `num_classes` defaults to the 24 static letters."""

    def __init__(self, num_classes: int = 24, input_size: int = 400, dropout: float = 0.25):
        super().__init__()
        if input_size != 400:
            raise ValueError("This paper-based implementation expects 400x400 inputs.")
        self.num_classes = num_classes
        # ReLU convolutional features; pooling after each block is an explicit assumption.
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )
        # 400 -> 398 -> 199 -> 197 -> 98 -> 96 -> 48 with valid convolutions.
        flattened_features = 128 * 48 * 48
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return (batch, num_classes) logits for (batch, 1, 400, 400) input."""
        if x.ndim != 4 or x.shape[1:] != (1, 400, 400):
            raise ValueError(f"Expected (batch, 1, 400, 400), got {tuple(x.shape)}")
        return self.classifier(self.features(x))


def load_best_model(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> tuple[ASLAlphabetCNN, dict]:
    """Load a saved best_model.pt from any local path, including a Drive sync folder.

    Returns the model in evaluation mode and its checkpoint metadata. The caller
    should apply the same grayscale, 400x400, [0, 1] preprocessing used in training.
    """
    path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise ValueError(f"Checkpoint is missing class_names metadata: {path}")
    model = ASLAlphabetCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint

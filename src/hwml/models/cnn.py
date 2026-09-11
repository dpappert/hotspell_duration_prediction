"""
CNN / 2D-input model architectures.

Collapses cnn_t7{a,b,c,d,lin,lr,mlp}_valid_{n,w,sw}.py -- 21 near-identical
scripts -- into 7 architecture classes + a registry, driven by
`scripts/validate_cnn.py --arch ...`.

Every architecture below was copied verbatim from its corresponding
original script. Full diffs between originals confirmed the *only*
differences across all 21 files were: (a) which architecture class is
used, (b) the region (f, c) values, and one path anomaly (see
`scripts/validate_cnn.py`). Hyperparameter grid and training loop were
byte-identical.
"""
import torch.nn as nn


class CNNa(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(16, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class CNNb(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        # Nonlinear MLP head (same size as MLP2D)
        self.classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class CNNc(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)


class CNNd(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        def dw_sep(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            )

        self.features = nn.Sequential(
            dw_sep(in_channels, 8),
            nn.MaxPool2d(2),
            dw_sep(8, 16),
            nn.MaxPool2d(2),
            dw_sep(16, 32),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class LinCNN(nn.Module):
    """No BatchNorm/ReLU between convs -- linear CNN, was `cnn_t7lin_*`."""

    def __init__(self, in_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1, bias=True),
            nn.Conv2d(8, 16, 3, padding=1, bias=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(16, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class LogReg2D(nn.Module):
    """Flattened logistic regression over the whole grid, was `cnn_t7lr_*`."""

    def __init__(self, in_channels, height, width):
        super().__init__()
        self.linear = nn.Linear(in_channels * height * width, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


class MLP2D(nn.Module):
    """Flattened MLP over the whole grid, was `cnn_t7mlp_*`."""

    def __init__(self, in_channels, height, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels * height * width, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# arch key (matches original filename suffix) -> (class, needs height/width, results subdir)
CNN_REGISTRY = {
    "a": (CNNa, False, "CNNa"),
    "b": (CNNb, False, "CNNb"),
    "c": (CNNc, False, "CNNc"),
    "d": (CNNd, False, "CNNd"),
    "lin": (LinCNN, False, "CNNlin"),
    "lr": (LogReg2D, True, "LogReg2D"),
    "mlp": (MLP2D, True, "MLP2D"),
}


def build_model(arch, in_channels, height=None, width=None):
    """Instantiate a registered architecture by its key."""
    if arch not in CNN_REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'; choose from {list(CNN_REGISTRY)}")
    cls, needs_hw, _ = CNN_REGISTRY[arch]
    if needs_hw:
        return cls(in_channels=in_channels, height=height, width=width)
    return cls(in_channels=in_channels)

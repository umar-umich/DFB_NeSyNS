"""
Frequency Feature Extractor
Supports: SRM + ResNet (custom), SPSL (DeepfakeBench registry), and future detectors.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class FrequencyFeatureExtractor(nn.Module):
    """
    Multi-backbone frequency feature extractor.

    Supported model names (config key: 'name'):
      - 'srm_resnet' : SRM filter bank + ResNet (default, self-contained)
      - 'spsl'       : Spatial-Phase Shallow Learning detector from DeepfakeBench.
                       Loads the registered SpslDetector, extracts its Xception
                       backbone, and computes phase maps on-the-fly — no external
                       preprocessing needed.

    Adding a new frequency detector in the future:
      1. Add a '_build_<name>()' method that sets self.backbone and self.backbone_dim.
      2. Add a '_forward_<name>()' method that returns (B, D) features.
      3. Register the name in _build_backbone() and forward().

    SPSL config example (config['foundation_models']['frequency']).
    Keys mirror the SPSL YAML directly so SpslDetector.__init__() receives
    exactly what it expects — no translation needed:

        name:          spsl
        pretrained:    ./training/pretrained/xception-b5690688.pth
        backbone_name: xception
        backbone_config:
            mode:        original
            num_classes: 2
            inc:         4        # ← Xception is built 4-channel from the start
            dropout:     false
        loss_func:     cross_entropy
        output_dim:    768        # projection target; backbone always outputs 2048
        freeze_backbone: true
    """

    def __init__(self, config):
        super().__init__()

        freq_config = config['foundation_models']['frequency']
        self.model_name = freq_config.get('name', 'srm_resnet')
        self.output_dim = freq_config['output_dim']
        self.freeze_backbone = freq_config.get('freeze_backbone', False)

        self._build_backbone(config)

        # Projection: align backbone dim to the desired output dim
        if self.backbone_dim != self.output_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.backbone_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
                nn.GELU(),
            )
        else:
            self.projection = nn.Identity()

        if self.freeze_backbone:
            self._freeze_backbone()

        print(f"Frequency Extractor: {self.model_name}")
        print(f"  Backbone dim: {self.backbone_dim} -> Output dim: {self.output_dim}")
        print(f"  Frozen: {self.freeze_backbone}")

    # ------------------------------------------------------------------
    # Backbone construction
    # ------------------------------------------------------------------

    def _build_backbone(self, config):
        if self.model_name == 'srm_resnet':
            self._build_srm_resnet(config)
        elif self.model_name == 'spsl':
            self._build_spsl(config)
        else:
            raise ValueError(f"Unknown frequency model: {self.model_name}")

    def _build_srm_resnet(self, config):
        """SRM filter bank + ResNet — fully self-contained, no external registry."""
        freq_config = config['foundation_models']['frequency']

        self.srm_conv = self._build_srm_layer(freq_config['srm_filters'])

        if freq_config.get('freeze_srm', False):
            for param in self.srm_conv.parameters():
                param.requires_grad = False

        resnet_depth = freq_config['resnet_depth']
        self.backbone = self._build_resnet(resnet_depth)

        # ResNet output dims before the FC layer
        resnet_dims = {18: 512, 34: 512, 50: 2048}
        self.backbone_dim = resnet_dims.get(resnet_depth, 512)

    def _build_spsl(self, config):
        """
        SPSL (Spatial-Phase Shallow Learning) from DeepfakeBench.

        Loads SpslDetector from the registry, then extracts its Xception backbone.
        The classification head is discarded — we only want the feature representations.

        SpslDetector.build_backbone() handles all the weight-loading and conv1
        averaging internally, so we pass freq_config (the flat SPSL-shaped sub-dict)
        directly and let it do its own initialization.

        The phase map is computed on-the-fly in _forward_spsl(), matching the
        original detector's phase_without_amplitude() exactly.
        """
        print("Loading SPSL detector from DeepfakeBench registry...")
        try:
            from detectors import DETECTOR
        except ImportError:
            raise ImportError(
                "DeepfakeBench 'detectors' registry not found. "
                "Ensure the DeepfakeBench package is installed and on PYTHONPATH."
            )

        # Pass the frequency sub-config directly: its keys (pretrained,
        # backbone_name, backbone_config, loss_func, …) exactly match what
        # SpslDetector.__init__() and build_backbone() expect.
        freq_config = config['foundation_models']['frequency']
        spsl_detector = DETECTOR['spsl'](freq_config)

        # Keep only the Xception backbone; loss_func and classifier head are unused.
        self.backbone = spsl_detector.backbone

        # Read the feature dim from Xception's last linear layer rather than
        # hardcoding it — survives any future backbone swap.
        # DeepfakeBench's Xception exposes this as backbone.last_linear.
        self.backbone_dim = self.backbone.last_linear.in_features

        print(f"  SPSL backbone: {freq_config['backbone_name']}"
              f"  (inc={freq_config['backbone_config']['inc']})")
        print(f"  SPSL backbone dim: {self.backbone_dim}")

    # ------------------------------------------------------------------
    # SRM helpers (used by srm_resnet only)
    # ------------------------------------------------------------------

    def _build_srm_layer(self, num_filters):
        srm_weights = self._get_srm_filters(num_filters)
        conv = nn.Conv2d(3, num_filters, kernel_size=5, padding=2, bias=False)
        conv.weight.data = torch.from_numpy(srm_weights).float()
        return conv

    def _get_srm_filters(self, num_filters):
        """
        Fixed high-pass SRM filters for manipulation artifact detection.
        Extend with the full 30-filter bank from steganalysis literature as needed.
        """
        f1 = np.array([
            [0,  0,  0,  0, 0],
            [0,  0,  0,  0, 0],
            [-1, -1,  4, -1, -1],
            [0,  0,  0,  0, 0],
            [0,  0,  0,  0, 0],
        ])
        f2 = np.array([
            [0, 0, -1, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0,  4, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0, -1, 0, 0],
        ])
        f3 = np.array([
            [0,  0,  0,  0, 0],
            [0, -1, -1, -1, 0],
            [0, -1,  8, -1, 0],
            [0, -1, -1, -1, 0],
            [0,  0,  0,  0, 0],
        ])
        base_filters = [f1, f2, f3]
        filters = []
        for i in range(num_filters):
            f = base_filters[i % len(base_filters)]
            filters.append(np.stack([f, f, f], axis=0))  # (3, 5, 5)
        return np.array(filters)  # (num_filters, 3, 5, 5)

    def _build_resnet(self, depth):
        if depth == 18:
            resnet = models.resnet18(pretrained=True)
        elif depth == 34:
            resnet = models.resnet34(pretrained=True)
        else:
            resnet = models.resnet50(pretrained=True)
        resnet.conv1 = nn.Conv2d(
            self.srm_conv.out_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False,
        )
        resnet.fc = nn.Identity()
        return resnet

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  ✓ Backbone frozen")

    # ------------------------------------------------------------------
    # Phase map (SPSL)
    # ------------------------------------------------------------------

    @staticmethod
    def _phase_without_amplitude(img: torch.Tensor) -> torch.Tensor:
        """
        Compute the phase-only reconstruction of an image, exactly as in
        SpslDetector.phase_without_amplitude().

        Args:
            img: (B, C, H, W) — RGB image tensor
        Returns:
            (B, 1, H, W) — real-valued phase reconstruction
        """
        gray = torch.mean(img, dim=1, keepdim=True)          # (B, 1, H, W)
        X = torch.fft.fftn(gray, dim=(-1, -2))
        phase_spectrum = torch.angle(X)
        reconstructed_X = torch.exp(1j * phase_spectrum)
        return torch.real(torch.fft.ifftn(reconstructed_X, dim=(-1, -2)))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — batch of images, already normalized
        Returns:
            (B, output_dim) — frequency features
        """
        if self.model_name == 'srm_resnet':
            features = self._forward_srm_resnet(x)
        elif self.model_name == 'spsl':
            features = self._forward_spsl(x)
        else:
            raise ValueError(f"Unknown frequency model: {self.model_name}")

        return self.projection(features)

    def _forward_srm_resnet(self, x: torch.Tensor) -> torch.Tensor:
        srm_out = self.srm_conv(x)
        return self.backbone(srm_out)   # (B, backbone_dim)

    def _forward_spsl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Replicates SpslDetector.features() without needing data_dict.

        SPSL concatenates the RGB image with its phase map along the channel
        dimension to form a 4-channel tensor, then passes it through Xception.
        
        Note: We need the full forward pass (features + pooling) to get the
        2048-dimensional vector, not just the spatial feature maps from .features().
        """
        phase = self._phase_without_amplitude(x)        # (B, 1, H, W)
        x_4ch = torch.cat([x, phase], dim=1)            # (B, 4, H, W)
        
        # Get spatial features
        spatial_features = self.backbone.features(x_4ch)  # (B, 2048, H', W')
        
        # Apply global average pooling to match the classifier's expected input
        # This replicates what happens in the full Xception forward pass
        pooled_features = torch.nn.functional.adaptive_avg_pool2d(
            spatial_features, (1, 1)
        ).view(spatial_features.size(0), -1)  # (B, 2048)
        
        return pooled_features
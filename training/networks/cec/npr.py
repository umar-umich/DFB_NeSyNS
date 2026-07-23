"""
networks/cec/npr.py
===================
NPR (Neighboring Pixel Relationships) — the second spectral instrument candidate.

NPR: Rethinking the Up-Sampling Operations in CNN-based Generative Network for
Generalizable Deepfake Detection, CVPR 2024 (chuangchuangtan). Detects the
periodic up-sampling artifact that generators leave: the input is replaced by
`x - interpolate(x, 0.5)` (the neighboring-pixel residual) before a truncated
ResNet-50 (conv1 + layer1 + layer2 → fc1 → one logit). A frequency-domain
signal, so it pairs with FreqNet on the spectral arm.

Architecture ported from NSG-VD/models/npr.py (itself from the chuangchuangtan
repo), reduced from that repo's 5-D video forward to the original 4-D image
forward the ProGAN checkpoint was trained with. NOT a GenD model — it loads
through its own path, not GenD's load_model.

Interface matches the GenD wrapper so CECInstrumentDetector treats it uniformly:
  model(tensor) -> HeadOutput(logits_labels)   # [real, fake], from one logit z
  model.get_preprocessing() -> (PIL.Image -> tensor)
p_fake = softmax([0, z])[:, 1] = sigmoid(z), the NPR convention (label 1 = fake).
"""
from __future__ import annotations

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

HeadOutput = namedtuple("HeadOutput", ["logits_labels"])

# NPR is evaluated on native-size images (AIGCDetectBenchmark: no_resize,
# no_crop) with ImageNet normalization. Our crops are a fixed even 256, so they
# batch without the odd-size trimming the repo needs for mixed-size benchmarks.
preprocessing_npr = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def conv3x3(inp, outp, stride=1):
    return nn.Conv2d(inp, outp, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(inp, outp, stride=1):
    return nn.Conv2d(inp, outp, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class NPRResNet(nn.Module):
    """Truncated ResNet (conv1 + layer1 + layer2) over the NPR residual.

    Matches the checkpoint: conv1 3x3 stride-2, Bottleneck layers [3, 4], fc1 512->1.
    """

    def __init__(self, block=Bottleneck, layers=(3, 4), num_classes=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    @staticmethod
    def _interpolate(img, factor):
        return F.interpolate(
            F.interpolate(img, scale_factor=factor, mode="nearest", recompute_scale_factor=True),
            scale_factor=1 / factor, mode="nearest", recompute_scale_factor=True,
        )

    def forward(self, x):
        # NPR residual (neighboring-pixel relationship), then the truncated resnet.
        npr = x - self._interpolate(x, 0.5)
        x = self.relu(self.bn1(self.conv1(npr * 2.0 / 3.0)))
        x = self.maxpool(x)
        x = self.layer2(self.layer1(x))
        x = self.avgpool(x).flatten(1)
        return self.fc1(x)  # (B, 1)


class NPRDetector(nn.Module):
    """NPR wrapped to the CEC instrument interface (frozen; never trained)."""

    def __init__(self):
        super().__init__()
        self.detector = NPRResNet()

    def forward(self, x):
        z = self.detector(x)              # (B, 1)
        if z.dim() == 1:
            z = z.unsqueeze(1)
        logits = torch.cat([torch.zeros_like(z), z], dim=1)  # softmax -> [1-σ(z), σ(z)]
        return HeadOutput(logits_labels=logits)

    def get_preprocessing(self):
        return preprocessing_npr

    def load_checkpoint(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.detector.load_state_dict(state, strict=False)
        return missing, unexpected


def build_npr_model(spec, device):
    """Build frozen NPR from a registration spec (`checkpoint` = absolute path)."""
    model = NPRDetector()
    missing, unexpected = model.load_checkpoint(spec["checkpoint"])
    if missing or unexpected:
        # NPR's truncated resnet should match the checkpoint exactly; a mismatch
        # means the wrong file or arch, which would silently produce garbage.
        raise RuntimeError(
            f"NPR checkpoint did not match the architecture: "
            f"{len(missing)} missing, {len(unexpected)} unexpected keys. "
            f"missing[:5]={list(missing)[:5]} unexpected[:5]={list(unexpected)[:5]}"
        )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, model.get_preprocessing()

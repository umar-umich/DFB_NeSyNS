import abc
from typing import Tuple

import torch
import torch.nn as nn


class AbstractDetector(nn.Module):
    """
    All deepfake detectors should subclass this class.
    """

    def __init__(self, *args, **kwargs):
        """
        config:   (dict)
            configurations for the model
        """
        super().__init__()

    def load_model(self, *args, **kwargs):
        """
        Returns the features from the backbone given the input data.
        """
        pass

    def set_input(self, *args, **kwargs):
        """
        Forward pass through the model, returning the prediction dictionary.
        """
        pass

    def get_predictions(self, *args, **kwargs):
        """
        Builds the backbone of the model.
        """
        pass

    def generate_outputs(self, *args, **kwargs):
        """
        Builds the backbone of the model.
        """
        pass


class AbstractModel:
    def __init(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(
        self, x: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.forward(x)
        probs = torch.nn.functional.softmax(x, dim=-1)
        score, pred = torch.max(probs, dim=1)
        return pred, score, probs

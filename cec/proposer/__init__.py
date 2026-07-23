"""CEC proposer: MLLM emits candidate claims (JSON), validator coerces to codebook.

  infer.py     Proposer.propose(image_path) -> raw JSON string (cached)
  validate.py  validate(text) -> ValidatedResponse (schema-valid?, decision, coerced claims)
"""

from .validate import Claim, ValidatedResponse, validate  # noqa: F401

__all__ = ["validate", "ValidatedResponse", "Claim", "Proposer"]


def __getattr__(name):
    # Proposer pulls in torch/transformers; import lazily so `validate` stays light.
    if name == "Proposer":
        from .infer import Proposer
        return Proposer
    raise AttributeError(name)

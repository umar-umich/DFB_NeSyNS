import importlib
import logging
import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import BACKBONE

logger = logging.getLogger(__name__)

# Optional backbones: (module, class name). Imported independently and skipped
# when their third-party dependencies are absent — same rationale as
# detectors/__init__.py: reaching one subpackage (e.g. networks.cec, which needs
# only torch) must not require every backbone's dependencies. With all
# dependencies installed this behaves exactly as the previous flat list did.
_OPTIONAL_BACKBONES = [
    ("xception", "Xception"),
    ("mesonet", "Meso4"),
    ("mesonet", "MesoInception4"),
    ("resnet34", "ResNet34"),
    ("efficientnetb4", "EfficientNetB4"),
    ("xception_sladd", "Xception_SLADD"),
]

unavailable_backbones = {}

for _module_name, _class_name in _OPTIONAL_BACKBONES:
    try:
        _module = importlib.import_module(f".{_module_name}", __name__)
        globals()[_class_name] = getattr(_module, _class_name)
    except ImportError as exc:
        unavailable_backbones[_class_name] = str(exc)

if unavailable_backbones:
    logger.info(
        "networks: %d of %d backbones unavailable in this environment (missing "
        "dependencies); the rest are registered normally. Skipped: %s",
        len(unavailable_backbones), len(_OPTIONAL_BACKBONES),
        ", ".join(sorted(unavailable_backbones)),
    )

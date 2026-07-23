import importlib
import logging
import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import DETECTOR

logger = logging.getLogger(__name__)

# Optional detectors: (module, class name).
#
# Each is imported independently and skipped if its third-party dependencies are
# absent. When every dependency is installed this behaves exactly as the previous
# flat list of imports did — same names, same registry, same order.
#
# Why: this package is imported to reach ONE detector, but the old flat list made
# that impossible unless every dependency of all ~30 was installed — video stacks
# (slowfast/fvcore), EfficientNet, dlib, and so on. The CEC audit runs in the
# `GenD` env, which deliberately holds only what the four frozen instruments need;
# installing the rest there would risk moving torch/numpy and silently renumbering
# the frozen anchors in CURRENT_STATE ledger item 6. Skipping a detector nobody
# asked for is free; a pinned dependency that shifts a score is not.
#
# A detector that fails to import is NOT silently ignored: it is logged, and
# asking DETECTOR for it raises a KeyError naming it.
_OPTIONAL_DETECTORS = [
    ("facexray_detector", "FaceXrayDetector"),
    ("xception_detector", "XceptionDetector"),
    ("efficientnetb4_detector", "EfficientDetector"),
    ("resnet34_detector", "ResnetDetector"),
    ("f3net_detector", "F3netDetector"),
    ("meso4_detector", "Meso4Detector"),
    ("meso4Inception_detector", "Meso4InceptionDetector"),
    ("spsl_detector", "SpslDetector"),
    ("core_detector", "CoreDetector"),
    ("capsule_net_detector", "CapsuleNetDetector"),
    ("srm_detector", "SRMDetector"),
    ("ucf_detector", "UCFDetector"),
    ("recce_detector", "RecceDetector"),
    ("fwa_detector", "FWADetector"),
    ("ffd_detector", "FFDDetector"),
    ("videomae_detector", "VideoMAEDetector"),
    ("clip_detector", "CLIPDetector"),
    ("timesformer_detector", "TimeSformerDetector"),
    ("xclip_detector", "XCLIPDetector"),
    ("sbi_detector", "SBIDetector"),
    ("ftcn_detector", "FTCNDetector"),
    ("i3d_detector", "I3DDetector"),
    ("altfreezing_detector", "AltFreezingDetector"),
    ("stil_detector", "STILDetector"),
    ("lsda_detector", "LSDADetector"),
    ("sladd_detector", "SLADDXceptionDetector"),
    ("pcl_xception_detector", "PCLXceptionDetector"),
    ("iid_detector", "IIDDetector"),
    ("lrl_detector", "LRLDetector"),
    ("rfm_detector", "RFMDetector"),
    ("uia_vit_detector", "UIAViTDetector"),
    ("multi_attention_detector", "MultiAttentionDetector"),
    ("sia_detector", "SIADetector"),
    ("tall_detector", "TALLDetector"),
    ("effort_detector", "EffortDetector"),
    ("nesy_defake_detector", "NeSyDeFakeHybridDetector"),
]

unavailable_detectors = {}

try:
    from .utils import slowfast
except ImportError as exc:  # video detectors only
    unavailable_detectors["utils.slowfast"] = str(exc)

for _module_name, _class_name in _OPTIONAL_DETECTORS:
    try:
        _module = importlib.import_module(f".{_module_name}", __name__)
        globals()[_class_name] = getattr(_module, _class_name)
    except ImportError as exc:
        unavailable_detectors[_module_name] = str(exc)

if unavailable_detectors:
    logger.info(
        "detectors: %d of %d unavailable in this environment (missing dependencies); "
        "the rest are registered normally. Skipped: %s",
        len(unavailable_detectors), len(_OPTIONAL_DETECTORS) + 1,
        ", ".join(sorted(unavailable_detectors)),
    )

# CEC instruments are NOT optional: they depend only on torch + the GenD
# checkout, and the audit cannot run without them. Let this raise.
from .cec_instrument_detector import (  # noqa: E402
    CECInstrumentDetector,
    CECEffortDetector,
    CECFsfmDetector,
    CECGendDetector,
    CECForadaDetector,
    CECFreqnetDetector,
    CECNprDetector,
)

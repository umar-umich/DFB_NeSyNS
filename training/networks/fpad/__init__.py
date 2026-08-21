"""FPAD — Face-Prior Adaptation Dynamics (WACV brief).

A frozen FS-VFM teacher and a LoRA-adapted student, read as a depth-resolved adaptation profile.
"""

from .teacher_student import (  # noqa: F401
    DEFAULT_LAYERS, FPAD_LORA, FPADTeacherStudent, build_teacher_student)
from .traj_head import (  # noqa: F401
    DirectEvidenceHead, TrajectoryEvidenceHead, build_traj_head)

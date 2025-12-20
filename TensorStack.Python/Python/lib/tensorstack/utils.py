import torch
from typing import Sequence, Optional
import numpy as np
import threading
from diffusers import (
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DDPMScheduler,
    DDIMScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DDPMWuerstchenScheduler,
    LCMScheduler,
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
    PNDMScheduler,
    HeunDiscreteScheduler,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverSDEScheduler
)

_SCHEDULER_MAP = {
    # Canonical
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "lms": LMSDiscreteScheduler,
    "euler": EulerDiscreteScheduler,
    "eulerancestral": EulerAncestralDiscreteScheduler,
    "kdpm2": KDPM2DiscreteScheduler,
    "kdpm2ancestral": KDPM2AncestralDiscreteScheduler,
    "ddpmwuerstchen": DDPMWuerstchenScheduler,
    "lcm": LCMScheduler,
    "flowmatcheulerdiscrete": FlowMatchEulerDiscreteScheduler,
    "flowmatchheundiscrete": FlowMatchHeunDiscreteScheduler,
    "pndm": PNDMScheduler,
    "heun": HeunDiscreteScheduler,
    "unipc": UniPCMultistepScheduler,
    "dpmm": DPMSolverMultistepScheduler,
    "dpms": DPMSolverSinglestepScheduler,
    "dpmsde": DPMSolverSDEScheduler,
}


def create_scheduler(name: str, *,config=None, **kwargs,):
    if not name:
        raise ValueError("Scheduler name must not be empty")

    key = name.lower().replace(" ", "").replace("-", "_")

    if key not in _SCHEDULER_MAP:
        raise ValueError(
            f"Unknown scheduler '{name}'. "
            f"Available: {sorted(_SCHEDULER_MAP.keys())}"
        )

    scheduler_cls = _SCHEDULER_MAP[key]

    if config is not None:
        # Best practice: preserve trained noise schedule
        if hasattr(config, "config"):
            return scheduler_cls.from_config(config.config, **kwargs)
        return scheduler_cls.from_config(config, **kwargs)

    return scheduler_cls(**kwargs)



def getDataType(dtype: str):
    if dtype == "float8_e5m2":
        return torch.float8_e5m2
    if dtype == "float8_e4m3fn":
        return torch.float8_e4m3fn
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float


def createTensor(
    inputData: Optional[Sequence[float]],
    inputShape: Optional[Sequence[int]],
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """
    Create a torch.Tensor from a flat float sequence + shape.

    Returns None if inputData or inputShape is None or empty.
    """

    if not inputData or not inputShape:
        return None

    shape = tuple(int(x) for x in inputShape)

    expected = 1
    for d in shape:
        expected *= d

    if len(inputData) != expected:
        raise ValueError(
            f"inputData length ({len(inputData)}) "
            f"does not match shape {shape} (expected {expected})"
        )

    np_array = np.asarray(inputData, dtype=np.float32).reshape(shape)
    return torch.from_numpy(np_array).to(device=device, dtype=dtype)


class MemoryStdout:
    def __init__(self, callback=None):
        self.callback = callback
        self._log_history = []
        self._lock = threading.Lock()

    def write(self, text):
        with self._lock:
            self._log_history.append(text)
        if self.callback:
            self.callback(text)

    def flush(self):
        pass  # no actual flushing needed here

    def get_log_history(self):
        with self._lock:
            logs_copy = self._log_history[:]
            self._log_history.clear()
        return logs_copy
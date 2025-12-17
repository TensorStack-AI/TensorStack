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
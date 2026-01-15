import gc
import ctypes
import ctypes.wintypes
import torch
import threading
import numpy as np
from PIL import Image
from typing import Sequence, Optional, List, Tuple, Union, Any, Dict
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
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
    DPMSolverMultistepInverseScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverSDEScheduler,
    DEISMultistepScheduler,
    EDMEulerScheduler,
    EDMDPMSolverMultistepScheduler,
    FlowMatchLCMScheduler,
    IPNDMScheduler
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
    "dpmminverse": DPMSolverMultistepInverseScheduler,
    "dpms": DPMSolverSinglestepScheduler,
    "dpmsde": DPMSolverSDEScheduler,
    "deism": DEISMultistepScheduler,
    "edm": EDMEulerScheduler,
    "edmm": EDMDPMSolverMultistepScheduler,
    "flowmatchlcm": FlowMatchLCMScheduler,
    "ipndm": IPNDMScheduler,
}


def create_scheduler(
    scheduler_name: str,
    scheduler_options: Dict[str, Any],
):
    scheduler_cls = _SCHEDULER_MAP[scheduler_name.lower()]

    # 0 = inf
    stmax = scheduler_options.get("s_tmax")
    if isinstance(stmax, (int, float)):
        scheduler_options["s_tmax"] = stmax if stmax > 0 else float("inf")

    # Defensive copy + drop None
    config = {k: v for k, v in scheduler_options.items() if v is not None}
   
    return scheduler_cls.from_config(config)

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


def configure_pipeline_memory(
    pipeline: Any,
    execution_device: str,
    pipelineOptions: Dict[str, Any],
) -> bool:
    """
    Configures memory offloading and VAE optimizations for a Diffusers pipeline.

    Returns:
        bool: True if any memory offload was enabled, False otherwise.
    """
    is_memory_offload = False
    is_full_offload_enabled = bool(pipelineOptions["is_full_offload_enabled"])
    is_model_offload_enabled = bool(pipelineOptions["is_model_offload_enabled"])
    is_vae_slicing_enabled = bool(pipelineOptions["is_vae_slicing_enabled"])
    is_vae_tiling_enabled = bool(pipelineOptions["is_vae_tiling_enabled"])

    # Memory offload
    if is_full_offload_enabled:
        is_memory_offload = True
        pipeline.enable_sequential_cpu_offload(device=execution_device)
    elif is_model_offload_enabled:
        is_memory_offload = True
        pipeline.enable_model_cpu_offload(device=execution_device)
    else:
        pipeline.to(execution_device)

    # VAE optimizations
    if hasattr(pipeline, "vae"):
        if is_vae_slicing_enabled:
            pipeline.vae.enable_slicing()
        if is_vae_tiling_enabled:
            pipeline.vae.enable_tiling()

    print(f"[configure_pipeline_memory]: is_memory_offload:{is_memory_offload}, is_full_offload_enabled:{is_full_offload_enabled}, is_model_offload_enabled:{is_model_offload_enabled}, is_vae_slicing_enabled:{is_vae_slicing_enabled}, is_vae_tiling_enabled:{is_vae_tiling_enabled}")
    return is_memory_offload


def tensorFromInput(
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


def imageFromInput(
    inputData: Optional[Sequence[float]],
    inputShape: Optional[Sequence[int]],
) -> Optional[Image.Image]:

    if not inputData or not inputShape:
        return None

    t = torch.tensor(inputData, dtype=torch.float32)
    t = t.view(*inputShape)
    t = t[0]
    t = (t + 1) / 2
    t = t.permute(1, 2, 0)
    t = (t.clamp(0, 1) * 255).to(torch.uint8)
    return Image.fromarray(t.numpy())


def prepare_images(
    lst: Optional[List[Tuple[Sequence[float], Sequence[int]]]]
) -> Optional[Union[Image.Image, List[Image.Image]]]:
    if not lst:
        return None

    def make_tensor(pair: Tuple[Sequence[float], Sequence[int]]):
        data, shape = pair
        return imageFromInput(data, shape)

    if len(lst) == 1:
        return make_tensor(lst[0])

    return [make_tensor(pair) for pair in lst]


def trim_memory(isMemoryOffload: bool):
    gc.collect()
    torch.cuda.empty_cache()

    if isMemoryOffload == True:
        SetProcessWorkingSetSizeEx = ctypes.windll.kernel32.SetProcessWorkingSetSizeEx
        SetProcessWorkingSetSizeEx.argtypes = [
            ctypes.wintypes.HANDLE,   # hProcess
            ctypes.c_size_t,          # dwMinimumWorkingSetSize
            ctypes.c_size_t,          # dwMaximumWorkingSetSize
            ctypes.wintypes.DWORD     # Flags
        ]
        SetProcessWorkingSetSizeEx.restype = ctypes.wintypes.BOOL
        h_process = ctypes.windll.kernel32.GetCurrentProcess()
        result = SetProcessWorkingSetSizeEx(
            h_process,
            ctypes.c_size_t(-1), # dwMinimumWorkingSetSize (disable)
            ctypes.c_size_t(-1), # dwMaximumWorkingSetSize (disable)
            0 # No special flags required for simple disable
        )


def isSingleFile(modelPath: str):
    return modelPath.lower().endswith((".safetensors", ".gguf"))


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
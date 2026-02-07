import os
import gc
import sys
import ctypes
import ctypes.wintypes
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import threading
import numpy as np
from tqdm import tqdm
import tensorstack.data_objects as DataObjects
from PIL import Image
from dataclasses import asdict
from huggingface_hub import hf_hub_download
from typing import Sequence, Optional, List, Tuple, Union, Any, Dict
from transformers import (
    AutoConfig
)
from diffusers.loaders import FromSingleFileMixin
from diffusers import (
    DiffusionPipeline, 
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
    IPNDMScheduler,
    CogVideoXDDIMScheduler, 
    CogVideoXDPMScheduler
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
    "cogvideoxddim": CogVideoXDDIMScheduler,
    "cogvideoxdpms": CogVideoXDPMScheduler
}


def create_scheduler(
    scheduler_name: str,
    scheduler_options: DataObjects.SchedulerOptions,
    scheduler_config: Dict[str, Any] = None
):
    scheduler_cls = _SCHEDULER_MAP[scheduler_name.lower()]
    config = dict(scheduler_config) if scheduler_config is not None else {}
    overrides = {k: v for k, v in asdict(scheduler_options).items() if v is not None}
    config.update(overrides)
    return scheduler_cls.from_config(config)


def configure_pipeline_memory(
    pipeline: Any,
    execution_device: str,
    config: DataObjects.PipelineConfig,
) -> bool:
    
    if config.memory_mode in("OffloadCPU", "LowMemDevice", "LowMemOffloadModel"):
        vae = getattr(pipeline, "vae", None)
        if callable(getattr(vae, "enable_tiling", None)):
            vae.enable_tiling()
        if callable(getattr(vae, "enable_slicing", None)):
            vae.enable_slicing()

    if config.memory_mode in("Device", "LowMemDevice"):
        pipeline.to(execution_device)

    elif config.memory_mode == "OffloadCPU":
        pipeline.enable_sequential_cpu_offload(device=execution_device)

    elif config.memory_mode in("OffloadModel", "LowMemOffloadModel"):
        pipeline.enable_model_cpu_offload(device=execution_device)

    return config.memory_mode in ("OffloadCPU", "OffloadModel", "LowMemOffloadModel")


def get_device_map(config: DataObjects.PipelineConfig):
    return "balanced" if config.memory_mode == "MultiDevice" else None


def get_pipeline_config(repo_id: str, cache_dir: str, secure_token: str) -> Dict[str, Optional[str]]:
    """
    Download all known pipeline component configs for a repo and return their local paths.
    Components not present will have value None.
    """

    config_paths: Dict[str, Optional[str]] = {}
    components = ["text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "transformer_2", "unet", "vae", "controlnet", "scheduler"]
    DiffusionPipeline.from_pretrained(repo_id, text_encoder=None, text_encoder_2=None, unet=None, vae=None, transformer=None, torch_dtype=None, cache_dir=cache_dir, token=secure_token)
    for comp in components:
        try:
            # All components: attempt to download config.json from the subfolder
            file_name = "config.json" if comp != "scheduler" else "scheduler_config.json"
            path = hf_hub_download(repo_id, f"{comp}/{file_name}", cache_dir=cache_dir, token=secure_token)
            if os.path.exists(path):
                config_paths[comp] = path
            else:
                config_paths[comp] = None
        except Exception:
            config_paths[comp] = None

    return config_paths


def load_lora_weights(pipeline: Any, config: DataObjects.PipelineConfig):
    pipeline.unload_lora_weights()
    if config.lora_adapters is not None:
        for lora in config.lora_adapters:
            pipeline.load_lora_weights(lora.path, weight_name=lora.weights, adapter_name=lora.name)


def set_lora_weights(pipeline: Any, config: DataObjects.PipelineOptions):
    if config.lora_options is not None:
        lora_map = { 
            opt.name: opt.strength
            for opt in config.lora_options
        }   
        names = list(lora_map.keys())
        weights = list(lora_map.values())
        pipeline.set_adapters(names, adapter_weights=weights)


def load_component(pipeline: FromSingleFileMixin, base_model_path: str, model_path: str, component_name: str, data_type: torch.dtype):
    try:
        components = ("scheduler", "tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2", "transformer", "transformer_2", "unet", "vae")
        skip_args = {c: None for c in components if c != component_name}
        pipe = pipeline.from_single_file(
            model_path,
            config=base_model_path,
            torch_dtype=data_type, 
            use_safetensors=True,
            local_files_only=True,
            **skip_args
        )

        return getattr(pipe, component_name, None)

    except Exception:
        return None
    

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


class ModelDownloadProgress:
    def __init__(self, total_models: int, total_per_model: int = 1000):
        self.total_per_model = total_per_model
        self.model_index: int = 0
        self.model_name: str = ""
        self.download_stats: Dict[str, Dict[str, float]] = {}  # filename -> {"downloaded": float, "total": float}
        self.total_models = total_models
        self._patched = False
        self.PatchTqdm()

    # --------------------
    # Public API
    # --------------------
    def Initialize(self, model_index: int, model_name: str):
        """Start tracking a new model. Previous model considered 100% complete."""
        if self.model_name:
            # Mark previous model as complete
            for fn in self.download_stats:
                if fn.startswith(self.model_name):
                    self.download_stats[fn]["downloaded"] = self.download_stats[fn]["total"]

        self.model_index = model_index
        self.model_name = model_name

        # Clear any previous files for this model
        for fn in list(self.download_stats.keys()):
            if fn.startswith(model_name):
                del self.download_stats[fn]

    def Update(self, filename: str, downloaded: float, total: float, speed: float):
        """Update a file's progress (MB)."""
        self.download_stats[filename] = {
            "downloaded": downloaded, 
            "total": total, 
            "speed": speed, 
            "model": self.model_name 
        }
        self._print_progress(filename)


    def Clear(self):
        """Clear all download tracking."""

        #print(f"[HUB_DOWNLOAD] | model | model | {self.total_per_model} | {self.total_per_model} | {self.total_per_model * self.total_models} | {self.total_per_model * self.total_models} | {0.00}")
        self.model_index = 0
        self.model_name = ""
        self.download_stats.clear()
     

    """Reset all download tracking."""
    def Reset(self, total_models: int):
        self.total_models = total_models
        self.Clear()

    # --------------------
    # Internal Methods
    # --------------------
    def _print_progress(self, filename: str):
        current_files = [x for x in self.download_stats.values() if x["model"] == self.model_name]
        if not current_files:
            avg_speed = 0.0
            model_progress = 0
        else:
            avg_speed = sum(x.get("speed", 0.0) for x in current_files)
            model_progress = sum(x["downloaded"] / max(x["total"], 0.001) for x in current_files) / len(current_files)

        scaled_model_progress = int(model_progress * self.total_per_model)
        overall_progress = self.model_index * self.total_per_model + scaled_model_progress
        max_progress = self.total_models * self.total_per_model

        print(f"[HUB_DOWNLOAD] | {self.model_name} | {filename} | {scaled_model_progress} | {self.total_per_model} | {overall_progress} | {max_progress} | {avg_speed:.2f}")

    # --------------------
    # TQDM Patch
    # --------------------
    def PatchTqdm(self):
        """Monkey-patch tqdm.update to feed progress automatically."""
        if self._patched:
            return  # only patch once

        original_update = tqdm.update
        progress_tracker = self

        def patched_update(self_tqdm, n=1):
            # Only process if total and desc exist
            if self_tqdm.n is not None and self_tqdm.total is not None and self_tqdm.desc:
                downloaded = self_tqdm.n / 1024 / 1024
                total_size = self_tqdm.total / 1024 / 1024
                speed = (
                    self_tqdm.format_dict.get("rate", 0.0) / 1024 / 1024
                    if self_tqdm.format_dict.get("rate")
                    else 0.001
                )

                # Extract model and filename
                model, filename = (self_tqdm.desc.split("/", 1) + [None])[:2]

                if model and filename and model == progress_tracker.model_name:
                    progress_tracker.Update(filename, downloaded, total_size, speed)
                elif model and progress_tracker.model_name == "control_net":
                    progress_tracker.Update(filename, downloaded, total_size, speed)

            return original_update(self_tqdm, n)

        tqdm.update = patched_update
        self._patched = True


def redirect_output():
    sys.stderr = MemoryStdout()
    sys.stdout = MemoryStdout()


def get_output() -> list[str]:
    return sys.stderr.get_log_history() + sys.stdout.get_log_history()

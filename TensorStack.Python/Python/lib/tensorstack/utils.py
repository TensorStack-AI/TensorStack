import os
import json
import gc
import sys
import ctypes
import ctypes.wintypes
import torch
import threading
from tqdm import tqdm
import tensorstack.data_objects as DataObjects
from PIL import Image
from dataclasses import asdict
from huggingface_hub import hf_hub_download, snapshot_download
from typing import Sequence, Optional, List, Tuple, Union, Any, Dict
from diffusers.loaders import FromSingleFileMixin
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
    DPMSolverSinglestepScheduler,
    DPMSolverSDEScheduler,
    DEISMultistepScheduler,
    EDMEulerScheduler,
    EDMDPMSolverMultistepScheduler,
    FlowMatchLCMScheduler,
    IPNDMScheduler,
    CogVideoXDDIMScheduler, 
    CogVideoXDPMScheduler,
    HeliosScheduler,
    HeliosDMDScheduler,
    TCDScheduler,
    SCMScheduler,
    SASolverScheduler,
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
    "flowmatcheuler": FlowMatchEulerDiscreteScheduler,
    "flowmatchheun": FlowMatchHeunDiscreteScheduler,
    "pndm": PNDMScheduler,
    "heun": HeunDiscreteScheduler,
    "unipcmultistep": UniPCMultistepScheduler,
    "dpmsolvermultistep": DPMSolverMultistepScheduler,
    "dpmsolversinglestep": DPMSolverSinglestepScheduler,
    "dpmsolversde": DPMSolverSDEScheduler,
    "deismultistep": DEISMultistepScheduler,
    "edmeuler": EDMEulerScheduler,
    "edmdpmsolvermultistep": EDMDPMSolverMultistepScheduler,
    "flowmatchlcm": FlowMatchLCMScheduler,
    "ipndm": IPNDMScheduler,
    "cogvideoxddim": CogVideoXDDIMScheduler,
    "cogvideoxdpms": CogVideoXDPMScheduler,
    "helios": HeliosScheduler,
    "heliosdmd": HeliosDMDScheduler,
    "tcd": TCDScheduler,
    "scm": SCMScheduler,
    "sasolver": SASolverScheduler
}


#------------------------------------------------
# Create a scheduler with the specifed options and configuration
#------------------------------------------------
def create_scheduler(
    scheduler_options: DataObjects.SchedulerOptions,
    scheduler_overrides: Dict[str, Any] = None
):
    scheduler_cls = _SCHEDULER_MAP[scheduler_options.Scheduler.lower()]
    options = {k: v for k, v in asdict(scheduler_options).items() if v is not None}
    overrides = dict(scheduler_overrides) if scheduler_overrides is not None else {}
    options.pop("Scheduler")
    options.update(overrides)

    print(f"[Scheduler]: {scheduler_cls.__name__}, {options}")
    return scheduler_cls.from_config(options)


#------------------------------------------------
# Get the model device_map
#------------------------------------------------
def get_device_map(config: DataObjects.PipelineConfig, execution_device: str):
    if config.memory_mode in("Balanced", "OffloadGPU"):
        return "cuda"
    elif config.memory_mode == "OffloadCPU":
        return "cpu"
    return None


#------------------------------------------------
# Get the pipeline device_map
#------------------------------------------------
def get_pipeline_device_map(config: DataObjects.PipelineConfig, execution_device: str):
    if config.memory_mode == "Balanced":
        return "balanced"
    elif config.memory_mode == "OffloadGPU":
        return "cuda"
    elif config.memory_mode == "OffloadCPU":
        return "cpu"
    return None


#------------------------------------------------
# Configure pipeline RAM/VRAM offloading
#------------------------------------------------
def configure_pipeline_memory(
    pipeline: Any,
    execution_device: str,
    config: DataObjects.PipelineConfig,
) -> bool:

    if config.memory_mode in("OffloadGPU"):
        optimize_pipeline(pipeline, config)
        pipeline.to(execution_device)

    elif config.memory_mode == "OffloadCPU":
        pipeline.enable_sequential_cpu_offload(device=execution_device)

    elif config.memory_mode in("OffloadModel"):
        pipeline.enable_model_cpu_offload(device=execution_device)

    return config.memory_mode in ("OffloadCPU", "OffloadModel")


#------------------------------------------------
# Configure VAE Tiling/Slicing
#------------------------------------------------
def configure_vae_memory(pipeline: Any, enable_tiling: bool, enable_slicing: bool):
    vae = getattr(pipeline, "vae", None)
    if not vae:
        return

    # Tiling: Processes the image in tiles to save VRAM on high-res images
    enable_t = getattr(vae, "enable_tiling", None)
    disable_t = getattr(vae, "disable_tiling", None)
    if callable(enable_t) and callable(disable_t):
        enable_t() if enable_tiling else disable_t()

    # Slicing: Processes the batch in slices
    enable_s = getattr(vae, "enable_slicing", None)
    disable_s = getattr(vae, "disable_slicing", None)
    if callable(enable_s) and callable(disable_s):
        enable_s() if enable_slicing else disable_s()


#------------------------------------------------
# Configure pipeline memory format NCHW or NHWC
#------------------------------------------------
def optimize_pipeline(pipeline: Any, config: DataObjects.PipelineConfig):
    if not config.is_optimize_channels_enabled:
        print(f"[Load] Optimize Channels Last: disabled")
        return

    if hasattr(pipeline, "unet"):
        print(f"[Load] Optimize Channels Last: channels_last")
        pipeline.vae.to(memory_format=torch.channels_last)
        pipeline.unet.to(memory_format=torch.channels_last)
        
    elif hasattr(pipeline, "transformer"):
        if config.process_type in ("TextToVideo", "ImageToVideo", "VideoToVideo"):
            #pipeline.vae.to(memory_format=torch.channels_last_3d)
            print(f"[Load] Optimize Channels Last: channels_last_3d")
            pipeline.transformer.to(memory_format=torch.channels_last_3d)
        else:
            print(f"[Load] Optimize Channels Last: channels_last")
            pipeline.vae.to(memory_format=torch.channels_last)
            pipeline.transformer.to(memory_format=torch.channels_last)


#------------------------------------------------
# Get the execution device
#------------------------------------------------
def get_execution_device(config: DataObjects.PipelineConfig):
    device_props = None
    device_index = None
    execution_device = None
    num_devices = torch.cuda.device_count()
    print(f"[Load] Request Device - Device: {config.device}, DeviceId: {config.device_id}, PCIBusId: {config.device_bus_id}")
    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        print(f"[Load] Found Device - Name: {props.name}, Index: {i}, PCIBusId: {props.pci_bus_id}, Arch: {getattr(props, 'gcnArchName', 'N/A')}")

        # Priority 1: Match by PCI Bus ID
        if config.device_bus_id > 0 and props.pci_bus_id == config.device_bus_id:
            device_index = i
            device_props = props

        # Priority 2: Fallback to Index if Bus ID is 0 or unavailable
        elif config.device_bus_id <= 0 and i == config.device_id:
            device_index = i
            device_props = props

    if device_props is not None:
        execution_device = f"{config.device}:{device_index}"
        print(f"[Load] Selected Device - Name: {device_props.name}, Index: {device_index}, PCIBusId: {device_props.pci_bus_id}, Arch: {getattr(device_props, 'gcnArchName', 'N/A')}, ExecutionDevice: {execution_device}")
        optimize_execution_device(config)
        return execution_device

    raise ValueError(f"Selected Device Not Found - Device: {config.device}, DeviceId: {config.device_id}, PCIBusId: {config.device_bus_id}")


#------------------------------------------------
# Set device specific optimizations
#------------------------------------------------
def optimize_execution_device(config: DataObjects.PipelineConfig):
    if not config.is_optimize_device_enabled:
        print(f"[Load] Optimize Device: disabled")
        return

    if not torch.cuda.is_available():
        return

    gpu_name = torch.cuda.get_device_name()
    major, minor = torch.cuda.get_device_capability()
    print(f"[Load] Optimize Device: {gpu_name} (Capability {major}.{minor})")

    # --- 1. SET MATMUL PRECISION ---
    if major >= 10: # Blackwell (RTX 4500)
        # Blackwell's 5th Gen Tensor Cores and TMEM path excel at "medium"
        # which utilizes the new FP4/FP6/FP8 pathways more aggressively.
        torch.set_float32_matmul_precision('medium')
    elif major >= 8: # Ampere/Ada (RTX 3090)
        # Ampere cards are better suited for "high" (TF32)
        torch.set_float32_matmul_precision('high')
    else:
        torch.set_float32_matmul_precision('highest')

    # --- 2. REDUCED PRECISION REDUCTION ---
    # This flag allows the GPU to use less precise math for sum-reductions
    # Blackwell has dedicated hardware for this that is significantly faster.
    if major >= 10:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    else:
        # On 3090 (8.6), this can sometimes cause "NaN" or black images in SDXL.
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # --- 3. CUDNN TF32 (For Convolutions) ---
    if major >= 8:
        torch.backends.cudnn.allow_tf32 = True





#------------------------------------------------
# Download/Load all config files needed to setup the pipeline, If files exists they are loaded form the cache
#------------------------------------------------
def get_pipeline_config(repo_id: str, cache_dir: str, secure_token: str) -> Dict[str, Optional[str]]:
    """
    Download all known pipeline component configs for a repo and return their local paths.
    Components not present will have value None.
    """
    config_paths: Dict[str, Optional[str]] = {}

    # Any Extra files
    allow_patterns = ["**/*.json", "*.json", "*.txt", "**/*.txt", "**/*.model", "**/*.jinja"]
    ignore_patterns = ["**/*.safetensors.index.json"]
    snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        token=secure_token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        user_agent="TensorStack-Diffuse"
    )

    known_components = [
        "text_encoder", 
        "text_encoder_2", 
        "text_encoder_3", 
        "transformer", 
        "transformer_2", 
        "unet", 
        "vae", 
        "vocoder", 
        "audio_vae", 
        "connectors", 
        "latent_upsampler", 
        "processor", 
        "image_processor", 
        "image_encoder", 
        "scheduler"
    ]
    for comp in known_components:
        try:
            # All components: attempt to download config.json from the subfolder
            file_name = "config.json" if comp != "scheduler" else "scheduler_config.json"
            path = hf_hub_download(
                repo_id, 
                f"{comp}/{file_name}", 
                cache_dir=cache_dir, 
                token=secure_token,
                user_agent="TensorStack-Diffuse"
            )
            if os.path.exists(path):
                config_paths[comp] = path
            else:
                config_paths[comp] = None
        except Exception:
            config_paths[comp] = None

    return config_paths


#------------------------------------------------
# Load the LoRA weights into the specified pipeline
#------------------------------------------------
def load_lora_weights(pipeline: Any, config: DataObjects.PipelineConfig):
    if not hasattr(pipeline, "load_lora_weights") or not hasattr(pipeline, "unload_lora_weights"):
        return

    pipeline.unload_lora_weights()
    if config.lora_adapters is not None:
        for lora in config.lora_adapters:
            pipeline.load_lora_weights(lora.path, weight_name=lora.weights, adapter_name=lora.name)


#------------------------------------------------
# Set the LoRA weights for inference
#------------------------------------------------
def set_lora_weights(pipeline: Any, config: DataObjects.PipelineOptions):
    if config.lora_options is not None:
        lora_map = { 
            opt.name: opt.strength
            for opt in config.lora_options
        }   
        names = list(lora_map.keys())
        weights = list(lora_map.values())
        pipeline.set_adapters(names, adapter_weights=weights)


#------------------------------------------------
# Try exctract and load an individual pipeline component from a single file
# If weights for the specified componenet do not exist None is returned
#------------------------------------------------
def load_component(pipeline: FromSingleFileMixin, base_model_path: str, model_path: str, component_name: str, data_type: torch.dtype, secure_token: str, device_map: Any):
    try:
        components = ("scheduler", "tokenizer", "tokenizer_2","tokenizer_3", "text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "transformer_2", "unet", "vae", "audio_vae", "vocoder", "connectors")
        skip_args = {c: None for c in components if c != component_name}
        pipe = pipeline.from_single_file(
            model_path,
            config=base_model_path,
            torch_dtype=data_type, 
            use_safetensors=True,
            low_cpu_mem_usage=True, 
            device_map=device_map,
            local_files_only=False,
            token=secure_token,
            **skip_args
        )

        return getattr(pipe, component_name, None)

    except Exception:
        return None


#------------------------------------------------
# Load a json file to dict
#------------------------------------------------
def load_json(file_path):
    """
    Safely loads a JSON file and returns a dictionary.
    Returns None or an empty dict if the file is missing or invalid.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return {}

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return {}


#------------------------------------------------
# Create a PIL image from and input buffer and shape
#------------------------------------------------
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


#------------------------------------------------
# Prepare the input image/video tensors
#------------------------------------------------
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


#------------------------------------------------
# Run garbage collection and empty cuda cache
#------------------------------------------------
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


#------------------------------------------------
# Is model path a single file
#------------------------------------------------
def isSingleFile(modelPath: str):
    return modelPath.lower().endswith((".safetensors", ".gguf"))


#------------------------------------------------
# Get length
#------------------------------------------------
def get_len(obj):
    if obj is None:
        return 0
    if hasattr(obj, '__len__'):
        return len(obj)
    return 1


#------------------------------------------------
# Redirect the modules stderr and stdout
#------------------------------------------------
def redirect_output():
    sys.stderr = MemoryStdout()
    sys.stdout = MemoryStdout()


#------------------------------------------------
# Get stderr and stdout log history
#------------------------------------------------
def get_output() -> list[str]:
    return sys.stderr.get_log_history() + sys.stdout.get_log_history()


#------------------------------------------------
# Helper class to intercept Stdout
#------------------------------------------------
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


#------------------------------------------------
# Helper class to parse diffusers progress to try get meaningful information
#------------------------------------------------
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

    def Update(self, key: str, filename: str, downloaded: float, total: float, speed: float):
        """Update a file's progress (MB)."""
        self.download_stats[key] = {
            "downloaded": downloaded, 
            "total": total, 
            "speed": speed, 
            "model": self.model_name 
        }
        self._print_progress(key, filename)


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
    def _print_progress(self, key:str, filename: str):
        current_files = [x for x in self.download_stats.values() if x["model"] == self.model_name]
        if not current_files:
            avg_speed = 0.0
            model_progress = 0
        else:
            avg_speed = sum(x.get("speed", 0.0) for x in current_files)
            model_progress = sum(x["downloaded"] / max(x["total"], 0.001) for x in current_files) / len(current_files)

        scaled_model_progress = int(model_progress * self.total_per_model)
        if scaled_model_progress <= 0 or filename == "Loading checkpoint shards":
            return

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
                #model, filename = (self_tqdm.desc.split("/", 1) + [None])[:2]
                model, *filename = self_tqdm.desc.split("/", 1)
                filename = filename[0] if filename else None

                if model and filename and model == progress_tracker.model_name:
                    progress_tracker.Update(self_tqdm.desc, filename, downloaded, total_size, speed)
                elif model and progress_tracker.model_name:
                    progress_tracker.Update(self_tqdm.desc, model, downloaded, total_size, speed)

            return original_update(self_tqdm, n)

        tqdm.update = patched_update
        self._patched = True


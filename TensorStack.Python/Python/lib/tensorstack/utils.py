import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_UPDATE_CHECK"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import json
import gc
import sys
import time
import ctypes
import ctypes.wintypes
import torch
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import tensorstack.data_objects as DataObjects
from tensorstack.enums import ProcessType, MemoryMode
from PIL import Image
from dataclasses import asdict
from collections.abc import Buffer
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
# Create model configuration
#------------------------------------------------
def get_model_config(file_path: str, config: DataObjects.PipelineConfig):
    template_path= Path(file_path).resolve().parent / "Templates" / config.template

    # Configs
    tokenizer_config = template_path / "tokenizer" / "tokenizer_config.json"
    tokenizer_2_config = template_path / "tokenizer_2" / "tokenizer_config.json"
    tokenizer_3_config = template_path / "tokenizer_3" / "tokenizer_config.json"
    text_encoder_config = template_path / "text_encoder" / "config.json"
    text_encoder_2_config = template_path / "text_encoder_2" / "config.json"
    text_encoder_3_config = template_path / "text_encoder_3" / "config.json"
    unet_config = template_path / "unet" / "config.json"
    transformer_config = template_path / "transformer" / "config.json"
    transformer_2_config = template_path / "transformer_2" / "config.json"
    vae_config = template_path / "vae" / "config.json"
    audio_vae_config = template_path / "audio_vae" / "config.json"
    vocoder_config = template_path / "vocoder" / "config.json"
    connectors_config = template_path / "connectors" / "config.json"
    latent_upsampler_config = template_path / "latent_upsampler" / "config.json"
    latent_upsampler_temporal_config = template_path / "latent_upsampler_temporal" / "config.json"
    condition_encoder_config = template_path / "condition_encoder" / "config.json"
    audio_tokenizer_config = template_path / "audio_tokenizer" / "config.json"
    audio_token_detokenizer_config = template_path / "audio_token_detokenizer" / "config.json"

    # Paths
    tokenizer_path = Path(config.checkpoint_config.text_encoder) if config.checkpoint_config.text_encoder else None
    tokenizer_2_path = Path(config.checkpoint_config.text_encoder_2) if config.checkpoint_config.text_encoder_2 else None
    tokenizer_3_path = Path(config.checkpoint_config.text_encoder_3) if config.checkpoint_config.text_encoder_3 else None
    text_encoder_path = Path(config.checkpoint_config.text_encoder) if config.checkpoint_config.text_encoder else None
    text_encoder_2_path = Path(config.checkpoint_config.text_encoder_2) if config.checkpoint_config.text_encoder_2 else None
    text_encoder_3_path = Path(config.checkpoint_config.text_encoder_3) if config.checkpoint_config.text_encoder_3 else None
    unet_path = Path(config.checkpoint_config.unet) if config.checkpoint_config.unet else None
    transformer_path = Path(config.checkpoint_config.transformer) if config.checkpoint_config.transformer else None
    transformer_2_path = Path(config.checkpoint_config.transformer_2) if config.checkpoint_config.transformer_2 else None
    vae_path = Path(config.checkpoint_config.vae) if config.checkpoint_config.vae else None
    audio_vae_path = Path(config.checkpoint_config.audio_vae) if config.checkpoint_config.audio_vae else None
    vocoder_path = Path(config.checkpoint_config.vocoder) if config.checkpoint_config.vocoder else None
    connectors_path = Path(config.checkpoint_config.connectors) if config.checkpoint_config.connectors else None
    latent_upsampler_path = Path(config.checkpoint_config.latent_upsampler) if config.checkpoint_config.latent_upsampler else None
    latent_upsampler_temporal_path = Path(config.checkpoint_config.latent_upsampler_temporal) if config.checkpoint_config.latent_upsampler_temporal else None
    condition_encoder_path = Path(config.checkpoint_config.condition_encoder) if config.checkpoint_config.condition_encoder else None
    audio_tokenizer_path = Path(config.checkpoint_config.audio_tokenizer) if config.checkpoint_config.audio_tokenizer else None
    audio_token_detokenizer_path = Path(config.checkpoint_config.audio_token_detokenizer) if config.checkpoint_config.audio_token_detokenizer else None
    single_file = transformer_path if transformer_path and transformer_path.is_file() else unet_path if unet_path and unet_path.is_file() else None

    _model_config = {
        "template": template_path,
        "single_file": single_file,

        "tokenizer": tokenizer_path,
        "tokenizer_config": tokenizer_config,
        "tokenizer_2": tokenizer_2_path,
        "tokenizer_2_config": tokenizer_2_config,
        "tokenizer_3": tokenizer_3_path,
        "tokenizer_3_config": tokenizer_3_config,
        "text_encoder": text_encoder_path,
        "text_encoder_config": text_encoder_config,
        "text_encoder_2": text_encoder_2_path,
        "text_encoder_2_config": text_encoder_2_config,
        "text_encoder_3": text_encoder_3_path,
        "text_encoder_3_config": text_encoder_3_config,
        "unet": unet_path,
        "unet_config": unet_config,
        "transformer": transformer_path,
        "transformer_config": transformer_config,
        "transformer_2": transformer_2_path,
        "transformer_2_config": transformer_2_config,
        "vae": vae_path,
        "vae_config": vae_config,
        "audio_vae": audio_vae_path,
        "audio_vae_config": audio_vae_config,
        "vocoder": vocoder_path,
        "vocoder_config": vocoder_config,
        "connectors": connectors_path,
        "connectors_config": connectors_config,
        "latent_upsampler": latent_upsampler_path,
        "latent_upsampler_config": latent_upsampler_config,
        "latent_upsampler_temporal": latent_upsampler_temporal_path,
        "latent_upsampler_temporal_config" : latent_upsampler_temporal_config,
        "condition_encoder": condition_encoder_path,
        "condition_encoder_config": condition_encoder_config,
        "audio_tokenizer": audio_tokenizer_path,
        "audio_tokenizer_config": audio_tokenizer_config,
        "audio_token_detokenizer": audio_token_detokenizer_path,
        "audio_token_detokenizer_config": audio_token_detokenizer_config,
    }

    info_1 = f"\n\tTemplate: {config.template} \n\tModelType: {config.model_type} \n\tModelPath: {config.model_path} \n\tTemplatePath: {template_path}"
    info_2 = f"\n\tTextEncoder: {text_encoder_path}\n\tTextEncoder2: {text_encoder_2_path}\n\tTextEncoder3: {text_encoder_3_path}\n\tUnet: {unet_path}\n\tTransformer: {transformer_path}\n\tTransformer2: {transformer_2_path}"
    info_3 = f"\n\tVae: {vae_path}\n\tAudioVae: {audio_vae_path}\n\tVocoder: {vocoder_path}\n\tConnectors: {connectors_path}\n\tLatentUpsampler: {latent_upsampler_path}\n\tLatentUpsamplerTemporal: {latent_upsampler_temporal_path}"
    info_4 = f"\n\tConditionEncoder: {condition_encoder_path}\n\tAudioTokenizer: {audio_tokenizer_path}\n\tAudioDetokenizer: {audio_token_detokenizer_path}"
    print(f"[Load] Initialize Model... \n[ {info_1}{info_2}{info_3}{info_4} \n]")
    return _model_config


#------------------------------------------------
# Try extract and load an individual pipeline component from a single file
# If weights for the specified componenet do not exist None is returned
#------------------------------------------------
def from_component(pipeline: FromSingleFileMixin, component_name: str, model_path: str, template_path: str,  device_map: Any, data_type: torch.dtype, quantization_config: Any = None):
    try:
        if not hasattr(pipeline, "from_single_file"):
            print(f"[Load] Loading Component not supported.")
            return None

        components = (
            "scheduler",
            "tokenizer",
            "tokenizer_2",
            "tokenizer_3",
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "unet",
            "transformer",
            "transformer_2",
            "vae",
            "audio_vae",
            "vocoder",
            "connectors",
            "latent_upsampler",
            "latent_upsampler_temporal",
            "condition_encoder",
            "audio_tokenizer",
            "audio_token_detokenizer"
        )
        skip_args = {c: None for c in components if c != component_name}
        pipe = pipeline.from_single_file(
            str(model_path),
            config=str(template_path),
            torch_dtype=data_type,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            device_map=device_map,
            local_files_only=True,
            quantization_config=quantization_config,
            **skip_args
        )

        return getattr(pipe, component_name, None)

    except Exception as e:
        print(f"[Load] Component not found")
        return None


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
    if config.memory_mode in(MemoryMode.Balanced, MemoryMode.OffloadGPU):
        return "cuda"
    elif config.memory_mode == MemoryMode.OffloadCPU:
        return None

    if config.is_device_quantization_enabled:
        return "cuda"

    return None


#------------------------------------------------
# Get the pipeline device_map
#------------------------------------------------
def get_pipeline_device_map(config: DataObjects.PipelineConfig, execution_device: str):
    if config.memory_mode == MemoryMode.Balanced:
        return "balanced"
    elif config.memory_mode == MemoryMode.OffloadGPU:
        return "cuda"
    elif config.memory_mode == MemoryMode.OffloadCPU:
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

    if config.memory_mode == MemoryMode.OffloadGPU:
        optimize_pipeline(pipeline, config)
        pipeline.to(execution_device)

    elif config.memory_mode == MemoryMode.OffloadCPU:
        pipeline.enable_sequential_cpu_offload(device=execution_device)

    elif config.memory_mode == MemoryMode.OffloadModel:
        pipeline.enable_model_cpu_offload(device=execution_device)

    return config.memory_mode in (MemoryMode.OffloadCPU, MemoryMode.OffloadModel)


#------------------------------------------------
# Configure VAE Tiling/Slicing
#------------------------------------------------
def configure_vae_memory(pipeline: Any, enable_tiling: bool, enable_slicing: bool):
    vae = getattr(pipeline, "vae", None)
    if not vae:
        return

    print(f"[Execute] Set VAE Memory, enable_tiling: {enable_tiling}, enable_slicing: {enable_slicing}")
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
        if config.process_type in (ProcessType.TextToVideo, ProcessType.ImageToVideo, ProcessType.VideoToVideo):
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
# Load the LoRA weights into the specified pipeline
#------------------------------------------------
def load_lora_weights(pipeline: Any, config: DataObjects.PipelineConfig):
    if not hasattr(pipeline, "load_lora_weights") or not hasattr(pipeline, "unload_lora_weights"):
        return

    pipeline.unload_lora_weights()
    if config.lora_adapters is not None:
        for lora in config.lora_adapters:
            print(f"[Load] Loading LoRA Adapter, Name: {lora.name}")
            pipeline.load_lora_weights(
                lora.path,
                weight_name=lora.weights,
                adapter_name=lora.name,
                local_files_only=True
            )


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
# Create an Audio Tensor from input buffer and shape
#------------------------------------------------
def audioFromInput(
    inputData: Optional[Sequence[float]],
    inputShape: Optional[Sequence[int]],
) -> Optional[torch.Tensor]:

    if not inputData or not inputShape:
        return None

    t = torch.tensor(inputData, dtype=torch.float32)
    t = t.view(*inputShape)
    return t


#------------------------------------------------
# Prepare the input audio tensors
#------------------------------------------------
def prepare_audio(
    lst: Optional[List[Tuple[Sequence[float], Sequence[int]]]]
) -> Optional[Union[torch.Tensor, List[torch.Tensor]]]:
    if not lst:
        return None

    def make_tensor(pair: Tuple[Sequence[float], Sequence[int]]):
        data, shape = pair
        return audioFromInput(data, shape)

    # Return a single tensor if list length is 1, otherwise a list
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
# Is model path a gguf file
#------------------------------------------------
def isGGUF(modelPath: Path):
    return modelPath.suffix == ".gguf"


#------------------------------------------------
# Get length
#------------------------------------------------
def get_len(obj):
    if obj is None:
        return 0

    # If it's a tensor, we treat it as ONE object regardless of dimensions
    if isinstance(obj, torch.Tensor):
        return 1

    # If it's a list or tuple of tensors, return the count of the list
    if isinstance(obj, (list, tuple)):
        return len(obj)

    # Fallback for other objects
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
    def __init__(self, key="", callback=None):
        self.callback = callback
        self._log_history = []
        self._lock = threading.Lock()

    def write(self, text):
        with self._lock:
            timestamp = datetime.now()
            self._log_history.append(f"{timestamp.isoformat()}|{text}")
        if self.callback:
            self.callback(text)

    def flush(self):
        pass  # no actual flushing needed here

    def isatty(self):
        return True

    def get_log_history(self):
        with self._lock:
            logs_copy = self._log_history[:]
            self._log_history.clear()
        return logs_copy


#------------------------------------------------
# Stopwatch class to handle time mesurements
#------------------------------------------------
class Stopwatch:
    def __init__(self):
        self._start_time = None
        self._step_elapsed = 0
        self._total_accumulated = 0
        self._is_running = False

    def start(self):
        if not self._is_running:
            self._start_time = time.perf_counter()
            self._is_running = True

    def stop(self):
        if self._is_running:
            duration = time.perf_counter() - self._start_time
            self._step_elapsed += duration
            self._total_accumulated += duration
            self._is_running = False

        return self.total_elapsed_ms

    def reset(self):
        """Resets the current step timer but keeps the total history."""
        elapsed = self.elapsed_ms
        was_running = self._is_running
        if was_running:
            self.stop()

        self._step_elapsed = 0
        if was_running:
            self.start()

        return elapsed

    @property
    def elapsed_ms(self):
        """Time for the CURRENT step only."""
        current_segment = 0
        if self._is_running:
            current_segment = time.perf_counter() - self._start_time
        return (self._step_elapsed + current_segment) * 1000

    @property
    def total_elapsed_ms(self):
        """Total time since the very first start()."""
        current_segment = 0
        if self._is_running:
            current_segment = time.perf_counter() - self._start_time
        return (self._total_accumulated + current_segment) * 1000

_notification_service = None
def create_services():
    global _notification_service
    _notification_service = NotificationService()

def notification_get():
    return _notification_service.get()

def notification_push(key: str, subkey: str, value: int = 0, maximum: int = 0, batchValue: int = 0, batchMaximum: int = 0, message: str = None, elapsed: float = 0, timestamp: datetime = datetime.now(), tensor: Buffer = []):
    return _notification_service.push(key= key, subkey= subkey, value= value, maximum= maximum, batchValue= batchValue, batchMaximum= batchMaximum, message=message, elapsed= elapsed, timestamp= timestamp, tensor= tensor)

#------------------------------------------------
# Helper class handle notifications
#------------------------------------------------
class NotificationService:
    def __init__(self):
        self._items = []
        self._lock = threading.Lock()

    def push(self, key: str, subkey: str, value: int = 0, maximum: int = 0, batchValue: int = 0, batchMaximum: int = 0, message: str = None, elapsed: float = 0, timestamp: datetime = datetime.now(), tensor: Buffer = []):
        with self._lock:
            self._items.append((f"{key}|{subkey}|{timestamp.isoformat()}|{elapsed}|{value}|{maximum}|{batchValue}|{batchMaximum}|{message}", np.ascontiguousarray(tensor)))

    def get(self):
        with self._lock:
            items_copy = self._items[:]
            self._items.clear()
        return items_copy

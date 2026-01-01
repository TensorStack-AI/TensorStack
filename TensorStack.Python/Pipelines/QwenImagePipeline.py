import os
import sys
import torch
import numpy as np
from threading import Event
from collections.abc import Buffer
from typing import Coroutine, Dict, Sequence, List, Tuple, Optional, Union
from diffusers import QwenImagePipeline, QwenImageImg2ImgPipeline, QwenImageEditPlusPipeline, QwenImageControlNetPipeline, QwenImageControlNetModel
from tensorstack.utils import MemoryStdout, create_scheduler, getDataType, imageFromInput, prepare_images, trim_memory
sys.stderr = MemoryStdout()

# Globals
_pipeline = None
_processType = None;
_step_latent = None
_generator = None
_isMemoryOffload = False
_cancel_event = Event()
_pipelineMap = {
    "TextToImage": QwenImagePipeline,
    "ImageToImage": QwenImageImg2ImgPipeline,
    "ImageEdit": QwenImageEditPlusPipeline,
    "ControlNetImage": QwenImageControlNetPipeline,
}

def load(
        modelName: str,
        processType: str,
        controlNet: str = None,
        device: str = "cuda",
        deviceId: int = 0,
        dataType: str = "bfloat16",
        variant: str = None,
        cacheDir: str = None,
        secureToken: str = None,
        isModelOffloadEnabled: bool = False,
        isFullOffloadEnabled: bool= False,
        isVaeSlicingEnabled: bool= False,
        isVaeTilingEnabled: bool = False,
        loraAdapters: Optional[List[Tuple[str, str, str]]] = None
    ) -> bool:
    global _pipeline, _generator, _processType, _isMemoryOffload

    # Reset
    _reset()

    # Load Pipeline
    dtype = getDataType(dataType)
    _processType = processType;
    options = {
        "torch_dtype": dtype,
        "cache_dir": cacheDir,
        "token": secureToken,
        "variant": variant,
    }

    #ControlNet
    if controlNet is not None:
        controlnetModel = QwenImageControlNetModel.from_pretrained(controlNet, torch_dtype=dtype)
        options.update({"controlnet": controlnetModel,})

    # Create pipeline
    pipeline = _pipelineMap[_processType]
    _pipeline = pipeline.from_pretrained(modelName, **options)

    #Lora Adapters
    if loraAdapters is not None:
        for adapter_path, weight_name, adapter_name in loraAdapters:
            _pipeline.load_lora_weights(adapter_path, weight_name=weight_name, adapter_name=adapter_name)

    # Device
    execution_device = torch.device(f"{device}:{deviceId}")
    if isFullOffloadEnabled:
        _isMemoryOffload = True
        _pipeline.enable_sequential_cpu_offload(device=execution_device)
    elif isModelOffloadEnabled:
        _isMemoryOffload = True
        _pipeline.enable_model_cpu_offload(device=execution_device)
    else:
        _pipeline.to(execution_device)

    # Memory
    if isVaeSlicingEnabled:
        _pipeline.vae.enable_slicing()
    if isVaeTilingEnabled:
        _pipeline.vae.enable_tiling()
    _generator = torch.Generator(device=execution_device)
    return True


def unload() -> bool:
    global _pipeline
    _pipeline.remove_all_hooks()
    _pipeline.maybe_free_model_hooks()
    if hasattr(_pipeline,"tokenizer"):
        del _pipeline.tokenizer
    if hasattr(_pipeline,"text_encoder"):
        del _pipeline.text_encoder
    if hasattr(_pipeline,"transformer"):
        del _pipeline.transformer
    if hasattr(_pipeline,"vae"):
        del _pipeline.vae
    del _pipeline

    # Cleanup
    trim_memory(_isMemoryOffload)
    return True


def generateCancel() -> None:
    _cancel_event.set()


def generate(
        prompt: str,
        negativePrompt: str,
        guidanceScale: float,
        guidanceScale2: float,
        steps: int,
        steps2: int,
        height: int,
        width: int,
        seed: int,
        scheduler: str,
        numFrames: int,
        shift: float,
        strength: float,
        controlScale: float,
        loraOptions: Optional[Dict[str, float]] = None,
        inputData: Optional[List[Tuple[Sequence[float],Sequence[int]]]] = None,
        controlNetData: Optional[List[Tuple[Sequence[float],Sequence[int]]]] = None,
    ) -> Buffer:

    # Reset
    _reset()

    #scheduler
    _pipeline.scheduler = create_scheduler(scheduler, config=_pipeline.scheduler)

    #Lora Adapters
    if loraOptions is not None:
        names = list(loraOptions.keys())
        weights = list(loraOptions.values())
        _pipeline.set_adapters(names, adapter_weights=weights)

    # Input Images
    image = prepare_images(inputData)
    control_image = prepare_images(controlNetData)

    # Pipeline Options
    options = {
        "prompt": prompt,
        "negative_prompt": negativePrompt,
        "height": height,
        "width": width,
        "generator": _generator.manual_seed(seed),
        "true_cfg_scale": guidanceScale,
        "guidance_scale": guidanceScale2,
        "num_inference_steps": steps,
        "output_type": "np",
        "callback_on_step_end": _progress_callback,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }

    if _processType in ("ImageToImage"):
        options.update({ "image": image })

    if _processType == "ImageToImage":
        options.update({ "strength": strength })

    if _processType in ("ImageToImage", "ImageEdit"):
        options.update({ "image": image,  "num_images_per_prompt": 1 })

    if _processType == "ControlNetImage":
        options.update({
            "controlnet_conditioning_scale": controlScale,
            "control_image": control_image
        })

    # Run Pipeline
    output = _pipeline(**options)[0]

    # (Batch, Channel, Height, Width)
    output = output.transpose(0, 3, 1, 2).astype(np.float32)

    # Cleanup
    trim_memory(_isMemoryOffload)
    return np.ascontiguousarray(output)


def getLogs() -> list[str]:
    return sys.stderr.get_log_history()


def getStepLatent() -> Buffer:
    return _step_latent


def _reset():
    _cancel_event.clear()


def _log(message: str):
    sys.stderr.write(message)


def _progress_callback(pipe, step: int, total_steps: int, info: Dict):
    global _step_latent
    if _cancel_event.is_set():
        pipe._interrupt = True
        raise Exception("Operation Canceled")

    latents = info.get("latents")
    if latents is not None:
        _step_latent = np.ascontiguousarray(latents.float().cpu())

    return info

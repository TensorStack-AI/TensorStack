import os
import gc
import sys
import torch
import numpy as np
from threading import Event
from collections.abc import Buffer
from typing import Coroutine, Dict, Sequence, List, Tuple, Optional
from diffusers import ChromaPipeline, ChromaImg2ImgPipeline
from tensorstack.utils import MemoryStdout, create_scheduler, getDataType, imageFromInput
sys.stderr = MemoryStdout()

# Globals
_pipeline = None
_processType = None;
_step_latent = None
_generator = None
_cancel_event = Event()

def load(
        modelName: str,
        processType: str,
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
    global _pipeline, _generator, _processType

    # Reset
    _reset()

    # Load Pipeline
    torch_dtype = getDataType(dataType)
    _processType = processType;
    if _processType == "TextToImage":
        _pipeline = ChromaPipeline.from_pretrained(
            modelName, 
            torch_dtype=torch_dtype,
            cache_dir = cacheDir,
            token = secureToken,
            variant=variant
        )
    elif _processType == "ImageToImage":
        _pipeline = ChromaImg2ImgPipeline.from_pretrained(
            modelName, 
            torch_dtype=torch_dtype,
            cache_dir = cacheDir,
            token = secureToken,
            variant=variant
        )

    #Lora Adapters
    if loraAdapters is not None:
        for adapter_path, weight_name, adapter_name in loraAdapters:
            _pipeline.load_lora_weights(adapter_path, weight_name=weight_name, adapter_name=adapter_name)

    # Device
    execution_device = torch.device(f"{device}:{deviceId}")
    if isFullOffloadEnabled:
        _pipeline.enable_sequential_cpu_offload(device=execution_device)
    elif isModelOffloadEnabled:
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
    torch.cuda.empty_cache()
    gc.collect()
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
        loraOptions: Optional[Dict[str, float]] = None,
        inputData: Optional[Sequence[float]] = None,
        inputShape: Optional[Sequence[int]] = None
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

    # Run Pipeline
    if _processType == "TextToImage":
        output = _pipeline(
            prompt = prompt,
            negative_prompt = negativePrompt,
            height = height,
            width = width,
            generator = _generator.manual_seed(seed),
            guidance_scale = guidanceScale,
            num_inference_steps = steps,
            output_type = "np",
            callback_on_step_end = _progress_callback,
            callback_on_step_end_tensor_inputs = ["latents"]
        )[0]
    elif _processType == "ImageToImage":
        output = _pipeline(
            image = imageFromInput(inputData, inputShape),
            strength = strength,
            prompt = prompt,
            negative_prompt = negativePrompt,
            height = height,
            width = width,
            generator = _generator.manual_seed(seed),
            guidance_scale = guidanceScale,
            num_inference_steps = steps,
            output_type = "np",
            callback_on_step_end = _progress_callback,
            callback_on_step_end_tensor_inputs = ["latents"]
        )[0]

    # (Batch, Channel, Height, Width)
    output = output.transpose(0, 3, 1, 2)
    output = output.astype(np.float32)
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
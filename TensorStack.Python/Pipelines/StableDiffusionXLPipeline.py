import os
import sys
import torch
import numpy as np
from threading import Event
from collections.abc import Buffer
from typing import Coroutine, Dict, Sequence, List, Tuple, Optional, Union
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel
from tensorstack.utils import MemoryStdout, create_scheduler, getDataType, imageFromInput, prepare_images, trim_memory
sys.stderr = MemoryStdout()

# Globals
_pipeline = None
_processType = None
_step_latent = None
_generator = None
_isMemoryOffload = False
_prompt_cache_key = None
_prompt_cache_value = None
_cancel_event = Event()
_pipelineMap = {
    "TextToImage": StableDiffusionXLPipeline,
    "ImageToImage": StableDiffusionXLImg2ImgPipeline,
    "ControlNetImage": StableDiffusionXLControlNetPipeline,
    "ControlNetImageToImage": StableDiffusionXLControlNetImg2ImgPipeline,
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

    # Pipeline Options
    dtype = getDataType(dataType)
    _processType = processType
    options = {
        "torch_dtype": dtype,
        "cache_dir": cacheDir,
        "token": secureToken,
        "variant": variant,
    }

    #ControlNet
    if controlNet is not None:
        controlnetModel = ControlNetModel.from_pretrained(controlNet, torch_dtype=dtype)
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
    global _pipeline, _prompt_cache_key, _prompt_cache_value
    _prompt_cache_key = None
    _prompt_cache_value = None
    _pipeline.remove_all_hooks()
    _pipeline.maybe_free_model_hooks()
    if hasattr(_pipeline,"tokenizer"):
        del _pipeline.tokenizer
    if hasattr(_pipeline,"tokenizer_2"):
        del _pipeline.tokenizer_2
    if hasattr(_pipeline,"text_encoder"):
        del _pipeline.text_encoder
    if hasattr(_pipeline,"text_encoder_2"):
        del _pipeline.text_encoder_2
    if hasattr(_pipeline,"unet"):
        del _pipeline.unet
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
    global _prompt_cache_key, _prompt_cache_value
    guidanceScale = float(guidanceScale)

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

    # Prompt Cache
    prompt_cache_key = (prompt, negativePrompt)
    if _prompt_cache_key != prompt_cache_key:
        with torch.no_grad():
            _prompt_cache_value = _pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=_pipeline._execution_device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=guidanceScale > 1,
                negative_prompt=negativePrompt,
                negative_prompt_2=negativePrompt
            )
            _prompt_cache_key = prompt_cache_key

    # Pipeline Options
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = _prompt_cache_value
    options = {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        "height": height,
        "width": width,
        "generator": _generator.manual_seed(seed),
        "guidance_scale": float(guidanceScale),
        "num_inference_steps": steps,
        "output_type": "np",
        "callback_on_step_end": _progress_callback,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }

    if _processType in ("ImageToImage","ControlNetImageToImage"):
        options.update({ "image": image, "strength": float(strength)})

    if _processType == "ControlNetImage":
        options.update({ "image": control_image })

    if _processType == "ControlNetImageToImage":
        options.update({ "control_image": control_image })

    if _processType in ("ControlNetImage", "ControlNetImageToImage"):
        options.update({
            "control_guidance_start": 0.0,
            "control_guidance_end": 1.0,
            "controlnet_conditioning_scale": float(controlScale)
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

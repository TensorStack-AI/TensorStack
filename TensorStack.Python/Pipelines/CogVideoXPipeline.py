import os
import sys
import torch
import numpy as np
from threading import Event
from collections.abc import Buffer
from typing import Coroutine, Dict, Sequence, List, Tuple, Optional, Union
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline, CogVideoXVideoToVideoPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from tensorstack.utils import MemoryStdout, create_scheduler, getDataType, tensorFromInput, prepare_images, trim_memory
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
    "TextToVideo": CogVideoXPipeline,
    "ImageToVideo": CogVideoXImageToVideoPipeline,
    "VideoToVideo": CogVideoXVideoToVideoPipeline,
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
    _processType = processType
    pipeline = _pipelineMap[_processType]
    _pipeline = pipeline.from_pretrained(
        modelName, 
        torch_dtype=dtype,
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
    global _prompt_cache_key, _prompt_cache_value
    guidanceScale = float(guidanceScale)

    # Reset
    _reset()

    # scheduler
    if scheduler == "ddim":
        _pipeline.scheduler = CogVideoXDDIMScheduler.from_config(_pipeline.scheduler.config)
    if scheduler == "ddpm":
        _pipeline.scheduler = CogVideoXDPMScheduler.from_config(_pipeline.scheduler.config)

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
                negative_prompt=negativePrompt,
                do_classifier_free_guidance=guidanceScale > 1,
                num_videos_per_prompt=1,
                device=_pipeline._execution_device,
                max_sequence_length=226
            )
            _prompt_cache_key = prompt_cache_key

    # Pipeline Options
    (prompt_embeds, negative_prompt_embeds) = _prompt_cache_value
    options = {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "height": height,
        "width": width,
        "generator": _generator.manual_seed(seed),
        "guidance_scale": guidanceScale,
        "num_inference_steps": steps,
        "num_frames": numFrames,
        "num_videos_per_prompt": 1,
        "output_type": "np",
        "callback_on_step_end": _progress_callback,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }
    if _processType == "ImageToVideo":
        options.update({ "image": image, "use_dynamic_cfg": True })

    # Run Pipeline
    output = _pipeline(**options)[0]

    # (Frames, Channel, Height, Width)
    output = output.transpose(0, 1, 4, 2, 3).squeeze(axis=0).astype(np.float32)

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
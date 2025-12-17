import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import gc
import sys
import torch
import numpy as np
from threading import Event
from collections.abc import Buffer
from typing import Coroutine, Dict
from diffusers import StableDiffusionXLPipeline
from tensorstack.utils import MemoryStdout, create_scheduler
sys.stderr = MemoryStdout()

# Globals
_pipeline = None
_step_latent = None
_generator = None
_cancel_event = Event()

def load(
        modelName: str, 
        isModelOffloadEnabled: bool = False,
        isFullOffloadEnabled: bool= False, 
        isVaeSlicingEnabled: bool= False, 
        isVaeTilingEnabled: bool = False, 
        device: str = "cuda",
        deviceId: int = 0,
        dataType: str = "bfloat16",
        variant: str = None
    ) -> bool:
    global _pipeline, _generator

    # Reset
    _reset()

    # Load Pipeline
    torch_dtype = _getDataType(dataType)
    _pipeline = StableDiffusionXLPipeline.from_pretrained(
        modelName, 
        torch_dtype=torch_dtype,
        cache_dir = None,
        token = None,
        variant=variant
    )

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
    torch.cuda.empty_cache()
    gc.collect()
    return True


def generateCancel() -> None:
    _cancel_event.set()


def generate(
        prompt: str,
        negativePrompt: str,
        guidanceScale: float,
        steps: int,
        height: int,
        width: int,
        seed: int,
        scheduler: str,
        numFrames: int,
        shift: float,
        flowShift: float
    ) -> Buffer:

    # Reset
    _reset()

    #scheduler
    _pipeline.scheduler = create_scheduler(scheduler, config=_pipeline.scheduler, use_karras_sigmas=True)

    # Run Pipeline
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



def _getDataType(dtype: str):
    if dtype == "float8":
        return torch.float16
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float



def _progress_callback(pipe, step: int, total_steps: int, info: Dict):
    global _step_latent
    if _cancel_event.is_set():
        pipe._interrupt = True
        raise Exception("Operation Canceled")

    latents = info.get("latents")
    if latents is not None:
        _step_latent = np.ascontiguousarray(latents.float().cpu())

    return info
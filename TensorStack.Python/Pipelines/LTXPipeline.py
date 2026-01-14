import os
import sys
import tensorstack.utils as utils
sys.stderr = utils.MemoryStdout()
sys.stdout = utils.MemoryStdout()

import torch
import numpy as np
from threading import Event
from collections.abc import Buffer
from typing import Coroutine, Dict, Sequence, List, Tuple, Optional, Union, Any
from diffusers import LTXPipeline, LTXImageToVideoPipeline, FlowMatchEulerDiscreteScheduler

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
    "TextToVideo": LTXPipeline,
    "ImageToVideo": LTXImageToVideoPipeline
}


def load(
        pipelineOptions: Dict[str, Any],
        loraAdapters: Optional[List[Tuple[str, str, str]]] = None
    ) -> bool:
    global _pipeline, _generator, _processType, _isMemoryOffload
    print(f"[PipelineOptions] {pipelineOptions}")
    _reset()

    # Pipeline Options
    modelName = pipelineOptions["path"]
    device = pipelineOptions["device"]
    deviceId = int(pipelineOptions["device_id"])
    dtype = utils.getDataType(pipelineOptions["data_type"])
    _processType = pipelineOptions["process_type"]
    options = {
        "torch_dtype": dtype,
        "cache_dir": pipelineOptions.get("cache_directory"),
        "token": pipelineOptions.get("secure_token"),
        "variant": pipelineOptions.get("variant"),
    }

    # #ControlNet Options
    # controlNet = pipelineOptions.get("control_net_path")
    # if controlNet is not None:
    #     controlnetModel = ControlNetModel.from_pretrained(controlNet, torch_dtype=dtype)
    #     options.update({"controlnet": controlnetModel,})

    # Create pipeline
    pipeline = _pipelineMap[_processType]
    is_single_file = utils.isSingleFile(modelName)
    _pipeline = (
        pipeline.from_single_file(modelName, **options)
        if is_single_file
        else pipeline.from_pretrained(modelName, **options)
    )

    #Lora Options
    if loraAdapters is not None:
        print("[LoraAdapters] ", loraAdapters)
        for adapter_path, weight_name, adapter_name in loraAdapters:
            _pipeline.load_lora_weights(adapter_path, weight_name=weight_name, adapter_name=adapter_name)

    # Device Options
    execution_device = torch.device(f"{device}:{deviceId}")
    _generator = torch.Generator(device=execution_device)
    _isMemoryOffload = utils.configure_pipeline_memory(
        _pipeline, 
        execution_device, 
        pipelineOptions
    )
    return True


def generate(
        inferenceOptions: Dict[str, Any],
        schedulerOptions: Dict[str, Any],
        loraOptions: Optional[Dict[str, float]] = None,
        inputData: Optional[List[Tuple[Sequence[float],Sequence[int]]]] = None,
        controlNetData: Optional[List[Tuple[Sequence[float],Sequence[int]]]] = None,
    ) -> Buffer:
    global _prompt_cache_key, _prompt_cache_value
    print(f"[InferenceOptions] {inferenceOptions}")
    print(f"[SchedulerOptions] {schedulerOptions}")
    _reset()

    # Options
    prompt = str(inferenceOptions.get("prompt"))
    negativePrompt = str(inferenceOptions.get("negative_prompt"))
    guidanceScale = float(inferenceOptions["guidance_scale"])
    steps = int(inferenceOptions["steps"])
    height = int(inferenceOptions["height"])
    width = int(inferenceOptions["width"])
    seed = int(inferenceOptions["seed"])
    scheduler = str(inferenceOptions["scheduler"])
    numFrames = int(inferenceOptions["frames"])
    strength = float(inferenceOptions["strength"])
    controlScale = float(inferenceOptions["control_net_scale"])

    # scheduler
    _pipeline.scheduler = utils.create_scheduler(scheduler, schedulerOptions)

    #Lora Adapters
    if loraOptions is not None:
        print(f"[LoraOptions] {loraOptions}")
        names = list(loraOptions.keys())
        weights = list(loraOptions.values())
        _pipeline.set_adapters(names, adapter_weights=weights)

    # Input Images
    image = utils.prepare_images(inputData)
    control_image = utils.prepare_images(controlNetData)

    # Pipeline Options
    options = {
        "prompt": prompt,
        "negative_prompt": negativePrompt,
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
        options.update({ "image": image })

    # Run Pipeline
    output = _pipeline(**options)[0]

    # (Frames, Channel, Height, Width)
    output = output.transpose(0, 1, 4, 2, 3).squeeze(axis=0).astype(np.float32)

    # Cleanup
    utils.trim_memory(_isMemoryOffload)
    return np.ascontiguousarray(output)


def generateCancel() -> None:
    _cancel_event.set()


def unload() -> bool:
    global _pipeline, _prompt_cache_key, _prompt_cache_value
    _prompt_cache_key = None
    _prompt_cache_value = None
    if _pipeline is not None:
        if hasattr(_pipeline, "remove_all_hooks"):
            _pipeline.remove_all_hooks()
        if hasattr(_pipeline, "maybe_free_model_hooks"):
            _pipeline.maybe_free_model_hooks()
        for name in ("tokenizer", "text_encoder", "transformer", "vae"):
            if hasattr(_pipeline, name):
                setattr(_pipeline, name, None)
        _pipeline = None
    utils.trim_memory(_isMemoryOffload)
    return True


def getLogs() -> list[str]:
    return sys.stderr.get_log_history() + sys.stdout.get_log_history()


def getStepLatent() -> Buffer:
    return _step_latent


def _reset():
    _cancel_event.clear()

def _progress_callback(pipe, step: int, total_steps: int, info: Dict):
    global _step_latent
    if _cancel_event.is_set():
        pipe._interrupt = True
        raise Exception("Operation Canceled")

    latents = info.get("latents")
    if latents is not None:
        _step_latent = np.ascontiguousarray(latents.float().cpu())

    return info
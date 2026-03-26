import tensorstack.utils as Utils
import tensorstack.data_objects as DataObjects
import tensorstack.quantization as Quantization
from tensorstack.enums import ProcessType, QuantTarget
Utils.redirect_output()
Utils.create_services()

import torch
import numpy as np
from threading import Event
from collections.abc import Buffer
from typing import Dict, Sequence, List, Tuple, Optional, Any
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
)

# Globals
_config = None
_pipeline = None
_processType = None
_pipeline_config = None
_execution_device = None
_device_map = None
_pipeline_device_map = None
_control_net_name = None
_control_net_cache = None
_generator = None
_isMemoryOffload = False
_prompt_cache_key = None
_prompt_cache_value = None
_progress_tracker: Utils.ModelDownloadProgress = None
_cancel_event = Event()
_stopwatch = None
_pipelineMap = {
    ProcessType.TextToImage: StableDiffusionXLPipeline,
    ProcessType.ImageToImage: StableDiffusionXLImg2ImgPipeline,
    ProcessType.ImageInpaint: StableDiffusionXLInpaintPipeline,
    ProcessType.ImageControlNet: StableDiffusionXLControlNetPipeline,
    ProcessType.ImageToImageControlNet: StableDiffusionXLControlNetImg2ImgPipeline,
}


#------------------------------------------------
# Initialize Pipeline
#------------------------------------------------
def initialize(config: DataObjects.PipelineConfig):
    global _progress_tracker, _pipeline_config, _device_map, _pipeline_device_map

    _progress_tracker = Utils.ModelDownloadProgress(total_models=get_model_count(config))
    _pipeline_config = Utils.get_pipeline_config(config.base_model_path, config.cache_directory, config.secure_token, config.is_offline_mode)
    _device_map = Utils.get_device_map(config, _execution_device)
    _pipeline_device_map = Utils.get_pipeline_device_map(config, _execution_device)
    return create_pipeline(config)


#------------------------------------------------
# Download Pipeline
#------------------------------------------------
def download(config_args: Dict[str, Any]):
    global _config, _progress_tracker, _pipeline_config, _device_map

    _device_map = "meta"
    _config = DataObjects.PipelineConfig(**config_args)
    _progress_tracker = Utils.ModelDownloadProgress(total_models=get_model_count(_config))
    _pipeline_config = Utils.get_pipeline_config(_config.base_model_path, _config.cache_directory, _config.secure_token, _config.is_offline_mode)
    create_pipeline(_config, True)
    return True


#------------------------------------------------
# Load Pipeline
#------------------------------------------------
def load(config_args: Dict[str, Any]) -> bool:
    global _config, _pipeline, _generator, _processType, _execution_device, _isMemoryOffload

    # Config
    _config = DataObjects.PipelineConfig(**config_args)
    _execution_device = Utils.get_execution_device(_config)
    _generator = torch.Generator(device=_execution_device)
    _processType = _config.process_type

    # Initialize Pipeline
    _pipeline = initialize(_config)

    # Load Lora
    Utils.load_lora_weights(_pipeline, _config)

    # Memory
    _isMemoryOffload = Utils.configure_pipeline_memory(_pipeline, _execution_device, _config)
    Utils.trim_memory(_isMemoryOffload)
    return True


#------------------------------------------------
# Reload Pipeline - ProcessType, LoraAdapters and ControlNet are the only options that can be modified
#------------------------------------------------
def reload(config_args: Dict[str, Any]) -> bool:
    global _config, _pipeline, _processType

    # Config
    _config = DataObjects.PipelineConfig(**config_args)
    _processType = _config.process_type
    _progress_tracker.Reset(total_models=get_model_count(_config))

    # Rebuild Pipeline
    _pipeline.unload_lora_weights()
    _pipeline = create_pipeline(_config)

    # Load Lora
    Utils.load_lora_weights(_pipeline, _config)

    # Memory
    Utils.configure_pipeline_memory(_pipeline, _execution_device, _config)
    Utils.trim_memory(_isMemoryOffload)
    return True


#------------------------------------------------
# Switch Pipeline - ProcessType
#------------------------------------------------
def switch(process_type: ProcessType) -> bool:
    global _pipeline, _processType

    # Switch Pipeline
    current = _processType
    _processType = process_type
    _pipeline = create_pipeline(_config)

    print(f"[Generate] Switched pipeline: {current} => {process_type}")
    return True


#------------------------------------------------
# Cancel Generation
#------------------------------------------------
def generateCancel() -> None:
    _cancel_event.set()


#------------------------------------------------
# Unload Pipline
#------------------------------------------------
def unload() -> bool:
    global _pipeline, _prompt_cache_key, _prompt_cache_value
    _pipeline = None
    _prompt_cache_key = None
    _prompt_cache_value = None
    Utils.trim_memory(_isMemoryOffload)
    return True


#------------------------------------------------
# Get the notifications
#------------------------------------------------
def getNotifications() -> list[(str, Buffer)]:
    return Utils.notification_get()


#------------------------------------------------
# Get the log entires
#------------------------------------------------
def getLogs() -> list[str]:
    return Utils.get_output()


#------------------------------------------------
# Diffusers pipeline callback to capture step artifacts
#------------------------------------------------
def _progress_callback(pipe, step: int, total_steps: int, info: Dict):
    if _cancel_event.is_set():
        pipe._interrupt = True
        raise Exception("Operation Canceled")

    steps = pipe._num_timesteps
    elapsed = _stopwatch.reset()
    step_latents = info.get("latents")
    step_latents = step_latents.float().cpu() if step_latents is not None else []
    Utils.notification_push(key="Generate", subkey="Step", value=step + 1, maximum=steps, elapsed=elapsed, tensor=step_latents)
    return info


#------------------------------------------------
# Get pipeline model count
#------------------------------------------------
def get_model_count(config: DataObjects.PipelineConfig):
    return 5 if config.control_net.name is not None else 4


#------------------------------------------------
# Generate Image/Video
#------------------------------------------------
def generate(
        inference_args: Dict[str, Any],
        input_tensors: Optional[List[Tuple[Sequence[float],Sequence[int]]]] = None,
        control_tensors: Optional[List[Tuple[Sequence[float],Sequence[int]]]] = None,
    ) -> Sequence[Buffer]:
    global _prompt_cache_key, _prompt_cache_value, _stopwatch
    _cancel_event.clear()
    _pipeline._interrupt = False
    _stopwatch = Utils.Stopwatch()
    _stopwatch.start()

    # Input Images
    images = Utils.prepare_images(input_tensors)
    image_count = Utils.get_len(images)
    control_images = Utils.prepare_images(control_tensors)
    control_image_count = Utils.get_len(control_images)
    print(f"[Generate] Input Received - Tensors: {image_count}, Control Tensors: {control_image_count}")

    # ControlNet
    if _processType == ProcessType.ImageControlNet and image_count > 0:
        switch(ProcessType.ImageToImageControlNet)
    elif _processType == ProcessType.ImageToImageControlNet and image_count == 0:
        switch(ProcessType.ImageControlNet)

    # Options
    options = DataObjects.PipelineOptions(**inference_args)

    # Scheduler
    _pipeline.scheduler = Utils.create_scheduler(options.scheduler_options)

    # AutoEncoder
    Utils.configure_vae_memory(_pipeline, options.enable_vae_tiling, options.enable_vae_slicing)

    # Lora Adapters
    Utils.set_lora_weights(_pipeline, options)

    # Notify
    Utils.notification_push(key="Generate", subkey="Initialize", elapsed=_stopwatch.reset())

    # Prompt Cache
    prompt_cache_key = (options.prompt, options.negative_prompt, options.guidance_scale > 1)
    if _prompt_cache_key != prompt_cache_key:
        print(f"[Generate] Encoding prompt")
        with torch.no_grad():
            _prompt_cache_value = _pipeline.encode_prompt(
                prompt=options.prompt,
                prompt_2=options.prompt,
                device=_pipeline._execution_device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=options.guidance_scale > 1,
                negative_prompt=options.negative_prompt,
                negative_prompt_2=options.negative_prompt
            )
            _prompt_cache_key = prompt_cache_key

    # Notify
    Utils.notification_push(key="Generate", subkey="Encode", elapsed=_stopwatch.reset())

    # Pipeline Options
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = _prompt_cache_value
    pipeline_options = {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        "height": options.height,
        "width": options.width,
        "generator": _generator.manual_seed(options.seed),
        "guidance_scale": options.guidance_scale,
        "num_inference_steps": options.steps,
        "output_type": "np",
        "callback_on_step_end": _progress_callback,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }

    if _processType in (ProcessType.ImageToImage, ProcessType.ImageToImageControlNet):
        pipeline_options.update({ "image": images, "strength": options.strength})

    if _processType == ProcessType.ImageInpaint:
        pipeline_options.update({ "image": images[0], "mask_image": images[1], "strength": options.strength})

    if _processType == ProcessType.ImageControlNet:
        pipeline_options.update({ "image": control_images })

    if _processType == ProcessType.ImageToImageControlNet:
        pipeline_options.update({ "control_image": control_images })

    if _processType in (ProcessType.ImageControlNet, ProcessType.ImageToImageControlNet):
        pipeline_options.update({
            "control_guidance_start": 0.0,
            "control_guidance_end": 1.0,
            "controlnet_conditioning_scale": options.control_net_scale
        })

    # Run Pipeline
    output = _pipeline(**pipeline_options)[0]

    # (Batch, Channel, Height, Width)
    output = output.transpose(0, 3, 1, 2).astype(np.float32)

    # Notify
    Utils.notification_push(key="Generate", subkey="Decode", elapsed = _stopwatch.reset())
    Utils.notification_push(key="Generate", subkey="Complete", elapsed = _stopwatch.stop())

    # Cleanup
    Utils.trim_memory(_isMemoryOffload)
    return [ np.ascontiguousarray(output) ]


#------------------------------------------------
# Create a new pipeline
#------------------------------------------------
def create_pipeline(config: DataObjects.PipelineConfig, download_only: bool = False):
    pipeline_kwargs = {
        "variant": config.variant,
        "token": config.secure_token,
        "cache_dir": config.cache_directory
    }

    # Load Models
    text_encoder = load_text_encoder(config, pipeline_kwargs)
    text_encoder_2 = load_text_encoder_2(config, pipeline_kwargs)
    unet = load_unet(config, pipeline_kwargs)
    vae = load_vae(config, pipeline_kwargs)
    control_net = load_control_net(config, pipeline_kwargs)
    if control_net is not None:
        pipeline_kwargs.update({"controlnet": control_net})

    _progress_tracker.Clear()
    if download_only:
        return None

    # Build Pipeline
    pipeline = _pipelineMap[_processType]
    return pipeline.from_pretrained(
        config.base_model_path,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        unet=unet,
        vae=vae,
        torch_dtype=config.data_type,
        device_map=_pipeline_device_map,
        local_files_only=True,
        low_cpu_mem_usage=True,
        **pipeline_kwargs
    )


#------------------------------------------------
# Load CLIPTextModel
#------------------------------------------------
def load_text_encoder(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):

    if _pipeline and _pipeline.text_encoder:
        print(f"[Load] Loading Cached TextEncoder")
        return _pipeline.text_encoder

    _progress_tracker.Initialize(0, "text_encoder")
    checkpoint = (
        config.checkpoint_config.text_encoder
        if config.checkpoint_config.text_encoder
        else config.checkpoint_config.single_file
    )
    if checkpoint:
        print(f"[Load] Loading Checkpoint TextEncoder, IsOffline: {config.is_offline_mode}")
        text_encoder = Utils.load_pipeline_component(config, StableDiffusionXLPipeline, "text_encoder", checkpoint, _device_map)
        if text_encoder:
            Utils.trim_memory(True)
            return text_encoder

    print(f"[Load] Loading Pretrained TextEncoder, IsOffline: {config.is_offline_mode}")
    text_encoder = CLIPTextModel.from_pretrained(
        "TensorStack/TextEncoder",
        subfolder="CLIP-VIT-L",
        config=_pipeline_config["text_encoder"],
        torch_dtype=config.data_type,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        local_files_only=config.is_offline_mode,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return text_encoder


#------------------------------------------------
# Load CLIPTextModelWithProjection
#------------------------------------------------
def load_text_encoder_2(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    if _pipeline and _pipeline.text_encoder_2:
        print(f"[Load] Loading Cached TextEncoder2")
        return _pipeline.text_encoder_2

    _progress_tracker.Initialize(1, "text_encoder_2")
    checkpoint = (
        config.checkpoint_config.text_encoder_2
        if config.checkpoint_config.text_encoder_2
        else config.checkpoint_config.single_file
    )
    if checkpoint:
        print(f"[Load] Loading Checkpoint TextEncoder2, IsOffline: {config.is_offline_mode}")
        text_encoder = Utils.load_pipeline_component(config, StableDiffusionXLPipeline, "text_encoder_2", checkpoint, _device_map)
        if text_encoder:
            Utils.trim_memory(True)
            return text_encoder

    print(f"[Load] Loading Pretrained TextEncoder2, IsOffline: {config.is_offline_mode}")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "TensorStack/TextEncoder",
        subfolder="CLIP-VIT-G",
        config=_pipeline_config["text_encoder_2"],
        torch_dtype=config.data_type,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        local_files_only=config.is_offline_mode,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return text_encoder


#------------------------------------------------
# Load UNet2DConditionModel
#------------------------------------------------
def load_unet(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):

    if _pipeline and _pipeline.unet:
        print(f"[Load] Loading Cached Unet")
        return _pipeline.unet

    _progress_tracker.Initialize(2, "unet")
    checkpoint = (
        config.checkpoint_config.transformer
        if config.checkpoint_config.transformer
        else config.checkpoint_config.single_file
    )
    if checkpoint:
        print(f"[Load] Loading Checkpoint Unet, IsOffline: {config.is_offline_mode}")
        is_gguf = Utils.isGGUF(checkpoint)
        unet = UNet2DConditionModel.from_single_file(
            checkpoint,
            config=_pipeline_config["unet"],
            torch_dtype=config.data_type,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            device_map=_device_map,
            local_files_only=config.is_offline_mode,
            token=config.secure_token,
            quantization_config=Quantization.auto_single_file_config(config, QuantTarget.TRANSFORMER, is_gguf)
        )

        Quantization.quantize_model(config, unet, is_gguf)
        Utils.trim_memory(True)
        return unet

    print(f"[Load] Loading Pretrained Unet, IsOffline: {config.is_offline_mode}")
    unet =  UNet2DConditionModel.from_pretrained(
        config.base_model_path,
        subfolder="unet",
        torch_dtype=config.data_type,
        quantization_config=Quantization.auto_pretrained_config(config, QuantTarget.TRANSFORMER),
        use_safetensors=True,
        low_cpu_mem_usage=True,
        local_files_only=config.is_offline_mode,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return unet


#------------------------------------------------
# Load AutoencoderKL
#------------------------------------------------
def load_vae(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):

    if _pipeline and _pipeline.vae:
        print(f"[Load] Loading Cached Vae")
        return _pipeline.vae

    _progress_tracker.Initialize(3, "vae")
    checkpoint = (
        config.checkpoint_config.vae
        if config.checkpoint_config.vae
        else config.checkpoint_config.single_file
    )
    if checkpoint:
        print(f"[Load] Loading Checkpoint Vae")
        auto_encoder = Utils.load_pipeline_component(config, StableDiffusionXLPipeline, "vae", checkpoint, _device_map)
        if auto_encoder:
            Utils.trim_memory(True)
            return auto_encoder

    print(f"[Load] Loading Pretrained Vae, IsOffline: {config.is_offline_mode}")
    auto_encoder = AutoencoderKL.from_pretrained(
        "TensorStack/AutoEncoder",
        subfolder="StableDiffusionXL",
        torch_dtype=config.data_type,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        local_files_only=config.is_offline_mode,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return auto_encoder


#------------------------------------------------
# Load ControlNetModel
#------------------------------------------------
def load_control_net(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    global _control_net_name, _control_net_cache

    if _control_net_cache and _control_net_name == config.control_net.name:
        print(f"[Load] Loading Cached ControlNet")
        return _control_net_cache

    if config.control_net.name is None:
        _control_net_name = None
        _control_net_cache = None
        return None

    print(f"[Load] Loading Pretrained ControlNet, IsOffline: {config.control_net.is_offline_mode}")
    _control_net_name = config.control_net.name
    _progress_tracker.Initialize(4, "control_net")
    _control_net_cache = ControlNetModel.from_pretrained(
        config.control_net.path,
        torch_dtype=config.data_type,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        local_files_only=config.control_net.is_offline_mode,
        device_map=_device_map,
    )
    return _control_net_cache


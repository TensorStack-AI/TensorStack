import tensorstack.utils as Utils
import tensorstack.export as Export
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
from transformers import Gemma3ForConditionalGeneration
from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    LTX2VideoTransformer3DModel,
    LTX2Pipeline,
    LTX2ImageToVideoPipeline
)
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

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
    ProcessType.TextToVideo: LTX2Pipeline,
    ProcessType.ImageToVideo: LTX2ImageToVideoPipeline
}


#------------------------------------------------
# Initialize Pipeline
#------------------------------------------------
def initialize(config: DataObjects.PipelineConfig):
    global _progress_tracker, _pipeline_config, _device_map, _pipeline_device_map

    _progress_tracker = Utils.ModelDownloadProgress(total_models=get_model_count(config))
    _pipeline_config = Utils.get_pipeline_config(config.base_model_path, config.cache_directory, config.secure_token)
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
    _pipeline_config = Utils.get_pipeline_config(_config.base_model_path, _config.cache_directory, _config.secure_token)
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
    return 6 if config.control_net.name is not None else 5


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

    # Options
    options = DataObjects.PipelineOptions(**inference_args)

    # Scheduler
    _pipeline.scheduler = Utils.create_scheduler(options.scheduler_options)

    # AutoEncoder
    Utils.configure_vae_memory(_pipeline, options.enable_vae_tiling, options.enable_vae_slicing)

    #Lora Adapters
    Utils.set_lora_weights(_pipeline, options)

    # Notify
    Utils.notification_push(key="Generate", subkey="Initialize", elapsed=_stopwatch.reset())

    # Prompt Cache
    prompt_cache_key = (options.prompt, options.negative_prompt, options.guidance_scale > 1.0)
    if _prompt_cache_key != prompt_cache_key:
        print(f"[Generate] Encoding prompt")
        with torch.no_grad():
            _prompt_cache_value = _pipeline.encode_prompt(
                prompt=options.prompt,
                negative_prompt=options.negative_prompt,
                do_classifier_free_guidance=options.guidance_scale > 1.0,
                num_videos_per_prompt=1
            )
            _prompt_cache_key = prompt_cache_key

    # Notify
    Utils.notification_push(key="Generate", subkey="Encode", elapsed=_stopwatch.reset())

    # Pipeline Options
    (prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask) = _prompt_cache_value
    pipeline_options = {
        "prompt_embeds": prompt_embeds,
        "prompt_attention_mask": prompt_attention_mask,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_attention_mask": negative_prompt_attention_mask,
        "height": options.height,
        "width": options.width,
        "generator": _generator.manual_seed(options.seed),
        "guidance_scale": options.guidance_scale,
        "num_inference_steps": options.steps,
        "num_frames": options.frames,
        "frame_rate": options.frame_rate,
        "num_videos_per_prompt": 1,
        "return_dict": False,
        "output_type": "np",
        "callback_on_step_end": _progress_callback,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }
    if _processType == ProcessType.ImageToVideo:
        pipeline_options.update({ "image": images })

    # Run Pipeline
    output_video, output_audio = _pipeline(**pipeline_options)

    # Notify
    Utils.notification_push(key="Generate", subkey="Decode", elapsed = _stopwatch.reset())

    # Export Video
    Export.encode_video(
        output_video.squeeze(),
        fps=options.frame_rate,
        output_path=options.temp_filename,
        audio=output_audio[0].float().cpu(),
        audio_sample_rate=_pipeline.vocoder.config.output_sampling_rate,  # should be 24000
    )

    # Notify
    Utils.notification_push(key="Generate", subkey="Export", elapsed = _stopwatch.reset())
    Utils.notification_push(key="Generate", subkey="Complete", elapsed = _stopwatch.stop())

    # Cleanup
    Utils.trim_memory(_isMemoryOffload)

    # (Frames, Channel, Height, Width)
    return []


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
    transformer = load_transformer(config, pipeline_kwargs)
    vae = load_vae_video(config, pipeline_kwargs)
    audio_vae = load_vae_audio(config, pipeline_kwargs)
    vocoder = load_vocoder(config, pipeline_kwargs)
    connectors = load_connectors(config, pipeline_kwargs)
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
        transformer=transformer,
        vae=vae,
        audio_vae=audio_vae,
        vocoder=vocoder,
        connectors=connectors,
        torch_dtype=config.data_type,
        device_map=_pipeline_device_map,
        local_files_only=True,
        low_cpu_mem_usage=True,
        **pipeline_kwargs
    )


#------------------------------------------------
# Load Gemma3ForConditionalGeneration
#------------------------------------------------
def load_text_encoder(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):

    if _pipeline and _pipeline.text_encoder:
        print(f"[Load] Loading Cached TextEncoder")
        return _pipeline.text_encoder

    _progress_tracker.Initialize(0, "text_encoder")
    checkpoint = config.checkpoint_config.text_encoder
    if checkpoint:
        print(f"[Load] Loading Checkpoint TextEncoder")
        is_gguf = Utils.isGGUF(checkpoint)
        text_encoder = Gemma3ForConditionalGeneration.from_single_file(
            checkpoint,
            config=_pipeline_config["text_encoder"],
            torch_dtype=config.data_type,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            device_map=_device_map,
            local_files_only=False,
            token=config.secure_token,
            quantization_config=Quantization.auto_single_file_config(config, QuantTarget.TEXT_ENCODER, is_gguf),
        )

        Quantization.quantize_model(config, text_encoder, is_gguf)
        Utils.trim_memory(True)
        return text_encoder

    print(f"[Load] Loading Pretrained TextEncoder")
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
        "TensorStack/TextEncoder",
        subfolder="Gemma-3-12B-IT",
        config=_pipeline_config["text_encoder"],
        torch_dtype=config.data_type,
        quantization_config=Quantization.auto_pretrained_config(config, QuantTarget.TEXT_ENCODER),
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return text_encoder


#------------------------------------------------
# Load LTX2VideoTransformer3DModel
#------------------------------------------------
def load_transformer(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):

    if _pipeline and _pipeline.transformer:
        print(f"[Load] Loading Cached Transformer")
        return _pipeline.transformer

    _progress_tracker.Initialize(1, "transformer")
    checkpoint = (
        config.checkpoint_config.transformer
        if config.checkpoint_config.transformer
        else config.checkpoint_config.single_file
    )
    if checkpoint:
        print(f"[Load] Loading Checkpoint Transformer")
        is_gguf = Utils.isGGUF(checkpoint)
        transformer = LTX2VideoTransformer3DModel.from_single_file(
            checkpoint,
            config=_pipeline_config["transformer"],
            torch_dtype=config.data_type,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            device_map=None,
            local_files_only=False,
            token=config.secure_token,
            quantization_config=Quantization.auto_single_file_config(config, QuantTarget.TRANSFORMER, is_gguf),
            output_loading_info=False
        )

        Quantization.quantize_model(config, transformer, is_gguf)
        Utils.trim_memory(True)
        return transformer

    print(f"[Load] Loading Pretrained Transformer")
    transformer = LTX2VideoTransformer3DModel.from_pretrained(
        config.base_model_path,
        subfolder="transformer",
        torch_dtype=config.data_type,
        quantization_config=Quantization.auto_pretrained_config(config, QuantTarget.TRANSFORMER),
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map=None,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return transformer


#------------------------------------------------
# Load AutoencoderKLLTX2Video
#------------------------------------------------
def load_vae_video(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):

    if _pipeline and _pipeline.vae:
        print(f"[Load] Loading Cached Vae")
        return _pipeline.vae

    _progress_tracker.Initialize(2, "vae")
    checkpoint = (
        config.checkpoint_config.vae
        if config.checkpoint_config.vae
        else config.checkpoint_config.single_file
    )
    if checkpoint:
        print(f"[Load] Loading Checkpoint Vae")
        auto_encoder = Utils.load_pipeline_component(config, LTX2Pipeline, "vae", checkpoint, _device_map)
        if auto_encoder:
            Utils.trim_memory(True)
            return auto_encoder

    print(f"[Load] Loading Pretrained Vae")
    auto_encoder = AutoencoderKLLTX2Video.from_pretrained(
        "TensorStack/AutoEncoder",
        subfolder="LTX2",
        torch_dtype=config.data_type,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return auto_encoder


#------------------------------------------------
# Load AutoencoderKLLTX2Audio
#------------------------------------------------
def load_vae_audio(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):

    if _pipeline and _pipeline.audio_vae:
        print(f"[Load] Loading Cached Audio Vae")
        return _pipeline.audio_vae

    _progress_tracker.Initialize(3, "audio_vae")
    checkpoint = (
        config.checkpoint_config.audio_vae
        if config.checkpoint_config.audio_vae
        else config.checkpoint_config.single_file
    )
    if checkpoint:
        print(f"[Load] Loading Checkpoint Audio Vae")
        auto_encoder = Utils.load_pipeline_component(config, LTX2Pipeline, "audio_vae", checkpoint, _device_map)
        if auto_encoder:
            Utils.trim_memory(True)
            return auto_encoder

    print(f"[Load] Loading Pretrained Audio Vae")
    auto_encoder = AutoencoderKLLTX2Audio.from_pretrained(
        config.base_model_path,
        subfolder="audio_vae",
        torch_dtype=config.data_type,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return auto_encoder


#------------------------------------------------
# Load Vocoder
#------------------------------------------------
def load_vocoder(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):

    if _pipeline and _pipeline.vocoder:
        print(f"[Load] Loading Cached Vocoder")
        return _pipeline.vocoder

    _progress_tracker.Initialize(4, "vocoder")
    checkpoint = (
        config.checkpoint_config.vocoder
        if config.checkpoint_config.vocoder
        else config.checkpoint_config.single_file
    )
    if checkpoint:
        print(f"[Load] Loading Checkpoint Vocoder")
        vocoder = Utils.load_pipeline_component(config, LTX2Pipeline, "vocoder", checkpoint, _device_map)
        if vocoder:
            Utils.trim_memory(True)
            return vocoder


    print(f"[Load] Loading Pretrained Vocoder")
    vocoder = LTX2Vocoder.from_pretrained(
        config.base_model_path,
        subfolder="vocoder",
        torch_dtype=config.data_type,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return vocoder


#------------------------------------------------
# Load Connectors
#------------------------------------------------
def load_connectors(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):

    if _pipeline and _pipeline.connectors:
        print(f"[Load] Loading Cached Connectors")
        return _pipeline.connectors

    _progress_tracker.Initialize(5, "connectors")
    checkpoint = (
        config.checkpoint_config.connectors
        if config.checkpoint_config.connectors
        else config.checkpoint_config.single_file
    )
    if checkpoint:
        print(f"[Load] Loading Checkpoint Connectors")
        connectors = Utils.load_pipeline_component(config, LTX2Pipeline, "connectors", checkpoint, _device_map)
        if connectors:
            Utils.trim_memory(True)
            return connectors


    print(f"[Load] Loading Pretrained Connectors")
    connectors = LTX2TextConnectors.from_pretrained(
        config.base_model_path,
        subfolder="connectors",
        torch_dtype=config.data_type,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return connectors


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

    # _control_net_name = config.control_net.name
    # _progress_tracker.Initialize(3, "control_net")
    # _control_net_cache = ControlNetModel.from_pretrained(
    #     config.control_net.path,
    #     torch_dtype=config.data_type,
    #     use_safetensors=True,
    #     low_cpu_mem_usage=True,
    #     device_map=_device_map,
    # )
    return None
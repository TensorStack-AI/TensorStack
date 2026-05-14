import tensorstack.utils as Utils
import tensorstack.data_objects as DataObjects
import tensorstack.quantization as Quantization
from tensorstack.enums import ProcessType, QuantTarget
Utils.redirect_output()
Utils.create_services()

import torch
import numpy as np
import soundfile as sf
from threading import Event
from collections.abc import Buffer
from typing import Dict, Sequence, List, Tuple, Optional, Any
from transformers import AutoModel
from diffusers import (
    AutoencoderOobleck,
    AceStepTransformer1DModel,
    AceStepConditionEncoder,
    AceStepPipeline
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
    ProcessType.TextToAudio: AceStepPipeline,
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

    if _config.lora_adapters is not None:
        print(f"[Download] Download Lora Adapter")
        _progress_tracker = Utils.ModelDownloadProgress(total_models=1)
        Utils.download_lora_weights(_config)
        return True
    elif _config.control_net.name is not None:
        print(f"[Download] Download ControlNet")
        _progress_tracker = Utils.ModelDownloadProgress(total_models=1)
        load_control_net(_config, None)
        return True

    print(f"[Download] Download Pipeline")
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

    # Input Audio
    audio = Utils.prepare_audio(input_tensors)
    audio_count = Utils.get_len(audio)
    print(f"[Generate] Input Received - Tensors: {audio_count}")

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

    # Notify
    Utils.notification_push(key="Generate", subkey="Encode", elapsed=_stopwatch.reset())

    # Pipeline Options
    pipeline_options = {
        "prompt": options.prompt,
        "lyrics": options.prompt2,
        "audio_duration": -1 if options.duration <=0 else options.duration,
        "vocal_language": options.language,
        "num_inference_steps": options.steps,
        "guidance_scale": options.guidance_scale,
        "shift": options.scheduler_options.shift,
        "max_text_length": options.max_length,
        "max_lyric_length": options.max_length2,
        "bpm": None if options.bpm == 0 else options.bpm,
        "keyscale": options.keyscale,
        "timesignature": options.time_signature,
        "task_type": options.task,
        "audio_cover_strength": options.strength,
        "generator": _generator.manual_seed(options.seed),
        "output_type": "np",
        "callback_on_step_end": _progress_callback,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }

    if audio_count == 1:
        #pipeline_options.update({ "src_audio": audio })
        pipeline_options.update({ "reference_audio": audio })

    # Run Pipeline
    output = _pipeline(**pipeline_options)[0]

    audio = output[0]  # (channels, samples), 48 kHz
    sf.write(options.temp_filename, audio.T, _pipeline.sample_rate)

    # Notify
    Utils.notification_push(key="Generate", subkey="Decode", elapsed = _stopwatch.reset())
    Utils.notification_push(key="Generate", subkey="Complete", elapsed = _stopwatch.stop())

    # Cleanup
    Utils.trim_memory(_isMemoryOffload)
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
    vae = load_vae(config, pipeline_kwargs)
    condition_encoder = load_condition_encoder(config, pipeline_kwargs)

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
        condition_encoder=condition_encoder,
        torch_dtype=config.data_type,
        device_map=_pipeline_device_map,
        local_files_only=True,
        low_cpu_mem_usage=True,
        **pipeline_kwargs
    )


#------------------------------------------------
# Load TextEncoder Qwen3Model
#------------------------------------------------
def load_text_encoder(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):

    if _pipeline and _pipeline.text_encoder:
        print(f"[Load] Loading Cached TextEncoder")
        return _pipeline.text_encoder

    _progress_tracker.Initialize(0, "text_encoder")

    print(f"[Load] Loading Pretrained TextEncoder, IsOffline: {config.is_offline_mode}")
    text_encoder = AutoModel.from_pretrained(
        config.base_model_path,
        subfolder="text_encoder",
        torch_dtype=config.data_type,
        quantization_config=Quantization.auto_pretrained_config(config, QuantTarget.TEXT_ENCODER),
        use_safetensors=True,
        low_cpu_mem_usage=True,
        local_files_only=config.is_offline_mode,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return text_encoder


#------------------------------------------------
# Load AceStepTransformer1DModel
#------------------------------------------------
def load_transformer(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):

    if _pipeline and _pipeline.transformer:
        print(f"[Load] Loading Cached Transformer")
        return _pipeline.transformer

    _progress_tracker.Initialize(1, "transformer")

    print(f"[Load] Loading Pretrained Transformer, IsOffline: {config.is_offline_mode}")
    transformer = AceStepTransformer1DModel.from_pretrained(
        config.base_model_path,
        subfolder="transformer",
        torch_dtype=config.data_type,
        quantization_config=Quantization.auto_pretrained_config(config, QuantTarget.TRANSFORMER),
        use_safetensors=True,
        low_cpu_mem_usage=True,
        local_files_only=config.is_offline_mode,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return transformer


#------------------------------------------------
# Load AutoencoderOobleck
#------------------------------------------------
def load_vae(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):

    if _pipeline and _pipeline.vae:
        print(f"[Load] Loading Cached Vae")
        return _pipeline.vae

    _progress_tracker.Initialize(2, "vae")

    print(f"[Load] Loading Pretrained Vae, IsOffline: {config.is_offline_mode}")
    auto_encoder = AutoencoderOobleck.from_pretrained(
        config.base_model_path,
        subfolder="vae",
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
# Load ConditionEncoder
#------------------------------------------------
def load_condition_encoder(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):

    if _pipeline and _pipeline.condition_encoder:
        print(f"[Load] Loading Cached ConditionEncoder")
        return _pipeline.condition_encoder

    _progress_tracker.Initialize(3, "condition_encoder")

    print(f"[Load] Loading Pretrained ConditionEncoder, IsOffline: {config.is_offline_mode}")
    condition_encoder = AceStepConditionEncoder.from_pretrained(
        config.base_model_path,
        subfolder="condition_encoder",
        torch_dtype=config.data_type,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        local_files_only=config.is_offline_mode,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return condition_encoder



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

    # print(f"[Load] Loading Pretrained ControlNet, IsOffline: {config.control_net.is_offline_mode}")
    # _control_net_name = config.control_net.name
    # _progress_tracker.Initialize(4, "control_net")
    # _control_net_cache = ControlNetModel.from_pretrained(
    #     config.control_net.path,
    #     torch_dtype=config.data_type,
    #     use_safetensors=True,
    #     low_cpu_mem_usage=True,
    #     local_files_only=config.control_net.is_offline_mode,
    #     device_map=_device_map,
    #     cache_dir=config.cache_directory,
    # )
    return None
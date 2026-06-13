import tensorstack.utils as Utils
import tensorstack.data_objects as DataObjects
import tensorstack.quantization as Quantization
from tensorstack.enums import ProcessType, QuantTarget
Utils.redirect_output()
Utils.create_services()

import torch
import numpy as np
from pathlib import Path
from threading import Event
from collections.abc import Buffer
from typing import Dict, Sequence, List, Tuple, Optional, Any
from transformers import TokenizersBackend, Mistral3Model
from diffusers import (
    AutoencoderKLFlux2,
    ErnieImageTransformer2DModel,
    ErnieImagePipeline
)

# Globals
_config = None
_model_config = None
_pipeline = None
_processType = None
_execution_device = None
_device_map = None
_pipeline_device_map = None
_control_net_name = None
_control_net_cache = None
_generator = None
_isMemoryOffload = False
_prompt_cache_key = None
_prompt_cache_value = None
_cancel_event = Event()
_stopwatch = None
_pipelineMap = {
    ProcessType.TextToImage: ErnieImagePipeline
}


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

    steps = len(pipe.scheduler.timesteps)
    elapsed = _stopwatch.reset()
    step_latents = info.get("latents")
    step_latents = step_latents.float().cpu() if step_latents is not None else []
    Utils.notification_push(key="Generate", subkey="Step", value=step + 1, maximum=steps, elapsed=elapsed, tensor=step_latents)
    return info


#------------------------------------------------
# Initialize Pipeline
#------------------------------------------------
def initialize(config: DataObjects.PipelineConfig):
    global _model_config, _device_map, _pipeline_device_map

    _device_map = Utils.get_device_map(config, _execution_device)
    _pipeline_device_map = Utils.get_pipeline_device_map(config, _execution_device)
    _model_config = Utils.get_model_config(__file__, config)
    return create_pipeline(config)


#------------------------------------------------
# Load TokenizersBackend
#------------------------------------------------
def load_tokenizer(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    if _pipeline and _pipeline.tokenizer:
        print(f"[Load] Loading Cached Tokenizer")
        return _pipeline.tokenizer

    tokenizer_path: Path = _model_config["tokenizer"]
    tokenizer_config: Path = _model_config["tokenizer_config"]

    # 1. Load from pretrained folder
    print(f"[Load] Loading Pretrained Tokenizer")
    tokenizer = TokenizersBackend.from_pretrained(
        tokenizer_path,
        #config=tokenizer_config,
        dtype=config.data_type,
        **pipeline_kwargs
    )
    return tokenizer


#------------------------------------------------
# Load Mistral3Model
#------------------------------------------------
def load_text_encoder(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    if _pipeline and _pipeline.text_encoder:
        print(f"[Load] Loading Cached TextEncoder")
        return _pipeline.text_encoder

    text_encoder_path: Path = _model_config["text_encoder"]
    text_encoder_config: Path = _model_config["text_encoder_config"]

    # 1. Load from pretrained folder
    print(f"[Load] Loading Pretrained TextEncoder")
    text_encoder = Mistral3Model.from_pretrained(
        text_encoder_path,
        config=text_encoder_config,
        dtype=config.data_type,
        device_map=_device_map,
        quantization_config=Quantization.auto_pretrained_config(config, QuantTarget.TEXT_ENCODER),
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return text_encoder


#------------------------------------------------
# Load ErnieImageTransformer2DModel
#------------------------------------------------
def load_transformer(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    if _pipeline and _pipeline.transformer:
        print(f"[Load] Loading Cached Transformer")
        return _pipeline.transformer

    transformer_path: Path = _model_config["transformer"]
    transformer_config: Path = _model_config["transformer_config"]

    # 1. Load from single file
    if transformer_path.is_file():
        is_gguf = Utils.isGGUF(transformer_path)
        print(f"[Load] Loading File Transformer")
        transformer =  ErnieImageTransformer2DModel.from_single_file(
            str(transformer_path),
            config=str(transformer_config),
            torch_dtype=config.data_type,
            device_map=_device_map,
            quantization_config=Quantization.auto_single_file_config(config, QuantTarget.TRANSFORMER, is_gguf),
            **pipeline_kwargs
        )
        Quantization.quantize_model(config, transformer, is_gguf)
        Utils.trim_memory(True)
        return transformer

    # 2. Load from pretrained folder
    print(f"[Load] Loading Pretrained Transformer")
    transformer =  ErnieImageTransformer2DModel.from_pretrained(
        str(transformer_path),
        torch_dtype=config.data_type,
        device_map=_device_map,
        quantization_config=Quantization.auto_pretrained_config(config, QuantTarget.TRANSFORMER),
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return transformer


#------------------------------------------------
# Load AutoencoderKLFlux2
#------------------------------------------------
def load_vae(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    if _pipeline and _pipeline.vae:
        print(f"[Load] Loading Cached Vae")
        return _pipeline.vae

    vae_path: Path = _model_config["vae"]
    vae_config: Path = _model_config["vae_config"]
    single_path: Path = _model_config["single_file"]
    template_path: Path  = _model_config["template"]

    # 1. Load from single file
    if vae_path.is_file():
        print(f"[Load] Loading SingleFile Vae")
        auto_encoder =  AutoencoderKLFlux2.from_single_file(
            str(vae_path),
            config=str(vae_config),
            torch_dtype=config.data_type,
            device_map=_device_map,
            **pipeline_kwargs
        )
        Utils.trim_memory(True)
        return auto_encoder

    # 2. Load component from single file
    if single_path and single_path.is_file():
        print(f"[Load] Loading Component Vae")
        auto_encoder = Utils.from_component(ErnieImagePipeline, "vae", single_path, template_path, _device_map, config.data_type)
        if auto_encoder:
            Utils.trim_memory(True)
            return auto_encoder

    # 3. Load from pretrained folder
    print(f"[Load] Loading Pretrained Vae")
    auto_encoder = AutoencoderKLFlux2.from_pretrained(
        str(vae_path),
        torch_dtype=config.data_type,
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

    # print(f"[Load] Loading Pretrained ControlNet")
    # _control_net_name = config.control_net.name
    # _control_net_cache = ControlNetModel.from_pretrained(
    #     config.control_net.path,
    #     torch_dtype=config.data_type,
    #     device_map=_device_map,
    #     **pipeline_kwargs
    # )
    return _control_net_cache


#------------------------------------------------
# Create a new pipeline
#------------------------------------------------
def create_pipeline(config: DataObjects.PipelineConfig):
    template_path: Path = _model_config["template"]
    pipeline_kwargs = {
        "variant": config.variant,
        "use_safetensors":True,
        "low_cpu_mem_usage":True,
        "local_files_only":True,
    }

    # Load Models
    tokenizer = load_tokenizer(config, pipeline_kwargs)
    text_encoder = load_text_encoder(config, pipeline_kwargs)
    transformer = load_transformer(config, pipeline_kwargs)
    vae = load_vae(config, pipeline_kwargs)
    control_net = load_control_net(config, pipeline_kwargs)
    if control_net is not None:
        pipeline_kwargs.update({"controlnet": control_net})

    # Build Pipeline
    pipeline = _pipelineMap[_processType]
    return pipeline.from_pretrained(
        template_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        pe=None,
        pe_tokenizer=None,
        torch_dtype=config.data_type,
        device_map=_pipeline_device_map,
        **pipeline_kwargs
    )


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

    # Lora Adapters
    Utils.set_lora_weights(_pipeline, options)

    # Notify
    Utils.notification_push(key="Generate", subkey="Initialize", elapsed=_stopwatch.reset())

    # Prompt Cache
    # None

    # Notify
    Utils.notification_push(key="Generate", subkey="Encode", elapsed=_stopwatch.reset())

    # Pipeline Options
    pipeline_options = {
        "prompt": options.prompt,
        "negative_prompt": options.negative_prompt,
        "height": options.height,
        "width": options.width,
        "generator": _generator.manual_seed(options.seed),
        "guidance_scale": options.guidance_scale,
        "num_inference_steps": options.steps,
        "use_pe": False,
        "output_type": "np",
        "callback_on_step_end": _progress_callback,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }

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
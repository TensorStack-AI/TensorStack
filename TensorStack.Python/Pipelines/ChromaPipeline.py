import sys
import tensorstack.utils as Utils
import tensorstack.data_objects as DataObjects
Utils.redirect_output()

import torch
import numpy as np
from threading import Event
from collections.abc import Buffer
from typing import Dict, Sequence, List, Tuple, Optional, Union, Any
from transformers import T5EncoderModel
from diffusers import ( 
    AutoencoderKL, 
    ChromaTransformer2DModel,
    ChromaPipeline, 
    ChromaImg2ImgPipeline,
)

# Globals
_pipeline = None
_processType = None
_step_latent = None
_generator = None
_isMemoryOffload = False
_prompt_cache_key = None
_prompt_cache_value = None
_progress_tracker: Utils.ModelDownloadProgress = None
_cancel_event = Event()
_pipelineMap = {
    "TextToImage": ChromaPipeline,
    "ImageToImage": ChromaImg2ImgPipeline,
}


def load(config_args: Dict[str, Any], lora_args: Optional[List[Dict[str, Any]]] = None) -> bool:
    global _pipeline, _generator, _processType, _isMemoryOffload

    # Config
    config = DataObjects.PipelineConfig(**config_args)
    lora_config = [DataObjects.LoraConfig(**cfg) for cfg in lora_args or []]
    _processType = config.process_type
   
    # Pipeline
    _pipeline = create_pipeline(config)

    #Lora Adapters
    if lora_config is not None:
        for lora in lora_config:
            print(f"[LoraAdapter] {lora}")
            _pipeline.load_lora_weights(lora.path, weight_name=lora.weights, adapter_name=lora.name)

    # Device
    execution_device = torch.device(f"{config.device}:{config.device_id}")
    _generator = torch.Generator(device=execution_device)
    _isMemoryOffload = Utils.configure_pipeline_memory(_pipeline, execution_device, config)
    Utils.trim_memory(_isMemoryOffload)
    return True


def generate(
        inference_args: Dict[str, Any],
        scheduler_args: Dict[str, Any],
        lora_args: Optional[Dict[str, float]] = None,
        input_tensors: Optional[List[Tuple[Sequence[float],Sequence[int]]]] = None,
        control_tensors: Optional[List[Tuple[Sequence[float],Sequence[int]]]] = None,
    ) -> Buffer:
    global _prompt_cache_key, _prompt_cache_value
    _cancel_event.clear()
    
    # Options
    options = DataObjects.PipelineOptions(**inference_args)
    scheduler_options = DataObjects.SchedulerOptions(**scheduler_args)

    #scheduler
    _pipeline.scheduler = Utils.create_scheduler(options.scheduler, scheduler_options, _pipeline.scheduler.config)

    #Lora Adapters
    if lora_args is not None:
        names = list(lora_args.keys())
        weights = list(lora_args.values())
        _pipeline.set_adapters(names, adapter_weights=weights)

    # Input Images
    image = Utils.prepare_images(input_tensors)
    control_image = Utils.prepare_images(control_tensors)

    # Prompt Cache
    # No caching implemented

    # Pipeline Options
    pipeline_options = {
        "prompt": options.prompt,
        "negative_prompt": options.negative_prompt,
        "height": options.height,
        "width": options.width,
        "generator": _generator.manual_seed(options.seed),
        "guidance_scale": options.guidance_scale,
        "num_inference_steps": options.steps,
        "output_type": "np",
        "callback_on_step_end": _progress_callback,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }
    if _processType == "ImageToImage":
        pipeline_options.update({ "image": image,"strength": options.strength })

    # Run Pipeline
    output = _pipeline(**pipeline_options)[0]

    # (Batch, Channel, Height, Width)
    output = output.transpose(0, 3, 1, 2).astype(np.float32)

    # Cleanup
    Utils.trim_memory(_isMemoryOffload)
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
    Utils.trim_memory(_isMemoryOffload)
    return True


def getLogs() -> list[str]:
    return Utils.get_output()


def getStepLatent() -> Buffer:
    return _step_latent


def _progress_callback(pipe, step: int, total_steps: int, info: Dict):
    global _step_latent
    if _cancel_event.is_set():
        pipe._interrupt = True
        raise Exception("Operation Canceled")

    latents = info.get("latents")
    if latents is not None:
        _step_latent = np.ascontiguousarray(latents.float().cpu())

    return info


def create_pipeline(config: DataObjects.PipelineConfig):
    global _progress_tracker
    _progress_tracker = Utils.ModelDownloadProgress(total_models=3)

    # Configuration
    pipeline_config = Utils.get_pipeline_config(config.base_model_path, config.cache_directory)
    quant_config_diffusers, uant_config_transformers = Utils.get_quantize_model_config(config.data_type, config.quant_data_type)
    pipeline_kwargs = { "variant": config.variant, "token": config.secure_token, "cache_dir": config.cache_directory }

    # Load Models
    text_encoder = load_text_encoder(config, pipeline_config, uant_config_transformers, pipeline_kwargs)
    transformer = load_transformer(config, pipeline_config, quant_config_diffusers, pipeline_kwargs)
    vae = load_vae(config, pipeline_config, quant_config_diffusers, pipeline_kwargs)
    _progress_tracker.Clear()

    # Build Pipeline
    device_map = Utils.get_device_map(config)
    pipeline = _pipelineMap[config.process_type]
    return pipeline.from_pretrained(
        config.base_model_path,
        text_encoder=text_encoder,
        transformer=transformer, 
        vae=vae, 
        torch_dtype=config.data_type,
        device_map=device_map,
        local_files_only=True,
        **pipeline_kwargs
    )


# T5EncoderModel 
def load_text_encoder(
        config: DataObjects.PipelineConfig, 
        pipeline_config: Dict[str, str], 
        quant_config: Any, 
        pipeline_kwargs: Dict[str, str]
    ):

    _progress_tracker.Initialize(0, "text_encoder")
    checkpoint_config = config.checkpoint_config
    if checkpoint_config.text_encoder_checkpoint is not None:
        text_encoder = T5EncoderModel.from_single_file(
            checkpoint_config.text_encoder_checkpoint, 
            config=pipeline_config["text_encoder"],
            torch_dtype=config.data_type, 
            use_safetensors=True, 
            local_files_only=True
        )
        Utils.quantize_model(text_encoder, config.quant_data_type)
        return text_encoder
    
    return T5EncoderModel.from_pretrained(
        config.base_model_path, 
        subfolder="text_encoder",
        torch_dtype=config.data_type, 
        quantization_config=quant_config, 
        use_safetensors=True,
        **pipeline_kwargs
    )


# ChromaTransformer2DModel
def load_transformer(
        config: DataObjects.PipelineConfig, 
        pipeline_config: Dict[str, str], 
        quant_config: Any, 
        pipeline_kwargs: Dict[str, str]
    ):

    _progress_tracker.Initialize(1, "transformer")
    checkpoint_config = config.checkpoint_config
    if checkpoint_config.model_checkpoint is not None:
        transformer = ChromaTransformer2DModel.from_single_file(
            checkpoint_config.model_checkpoint, 
            config=pipeline_config["transformer"],
            torch_dtype=config.data_type, 
            use_safetensors=True, 
            local_files_only=True
        )
        Utils.quantize_model(transformer, config.quant_data_type)
        return transformer
    
    return ChromaTransformer2DModel.from_pretrained(
        config.base_model_path, 
        subfolder="transformer", 
        torch_dtype=config.data_type, 
        quantization_config=quant_config, 
        use_safetensors=True,
        **pipeline_kwargs
    )


# AutoencoderKL
def load_vae(
        config: DataObjects.PipelineConfig, 
        pipeline_config: Dict[str, str], 
        quant_config: Any, 
        pipeline_kwargs: Dict[str, str]
    ):

    _progress_tracker.Initialize(2, "vae")
    checkpoint_config = config.checkpoint_config
    if checkpoint_config.vae_checkpoint is not None:
        return AutoencoderKL.from_single_file(
            checkpoint_config.vae_checkpoint, 
            config=pipeline_config["vae"],
            torch_dtype=config.data_type, 
            use_safetensors=True,
            local_files_only=True
        )
    
    return AutoencoderKL.from_pretrained(
        config.base_model_path, 
        subfolder="vae", 
        torch_dtype=config.data_type, 
        use_safetensors=True,
        **pipeline_kwargs
    )


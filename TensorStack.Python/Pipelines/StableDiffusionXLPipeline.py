import sys
import tensorstack.utils as Utils
import tensorstack.data_objects as DataObjects
import tensorstack.quantization as Quantization
Utils.redirect_output()

import torch
import numpy as np
from threading import Event
from collections.abc import Buffer
from typing import Dict, Sequence, List, Tuple, Optional, Union, Any
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
_pipeline = None
_processType = None
_pipeline_config = None
_quant_config_diffusers = None
_quant_config_transformers = None
_execution_device = None
_device_map = None
_control_net_path = None
_control_net_cache = None
_step_latent = None
_generator = None
_isMemoryOffload = False
_prompt_cache_key = None
_prompt_cache_value = None
_progress_tracker: Utils.ModelDownloadProgress = None
_cancel_event = Event()
_pipelineMap = {
    "TextToImage": StableDiffusionXLPipeline,
    "ImageToImage": StableDiffusionXLImg2ImgPipeline,
    "ImageInpaint": StableDiffusionXLInpaintPipeline,
    "ControlNetImage": StableDiffusionXLControlNetPipeline,
    "ControlNetImageToImage": StableDiffusionXLControlNetImg2ImgPipeline,
}


#------------------------------------------------
# Initialize Pipeline
#------------------------------------------------
def initialize(config: DataObjects.PipelineConfig):
    global _progress_tracker, _pipeline_config,  _quant_config_diffusers, _quant_config_transformers, _device_map

    _progress_tracker = Utils.ModelDownloadProgress(total_models=5 if config.control_net_path is not None else 4)
    _pipeline_config = Utils.get_pipeline_config(config.base_model_path, config.cache_directory, config.secure_token)
    _quant_config_diffusers, _quant_config_transformers = Quantization.get_quantize_model_config(config.data_type, config.quant_data_type, config.memory_mode)
    _device_map = Utils.get_device_map(config)
    return create_pipeline(config)


#------------------------------------------------
# Load Pipeline
#------------------------------------------------
def load(config_args: Dict[str, Any]) -> bool:
    global _pipeline, _generator, _processType, _execution_device, _isMemoryOffload

    # Config
    config = DataObjects.PipelineConfig(**config_args)
    _processType = config.process_type

    # Initialize Pipeline
    _pipeline = initialize(config)

    # Load Lora
    Utils.load_lora_weights(_pipeline, config)

    # Memory
    _execution_device = torch.device(f"{config.device}:{config.device_id}")
    _generator = torch.Generator(device=_execution_device)
    _isMemoryOffload = Utils.configure_pipeline_memory(_pipeline, _execution_device, config)
    Utils.trim_memory(_isMemoryOffload)
    return True
    

#------------------------------------------------
# Reload Pipeline - ProcessType, LoraAdapters and ControlNet are the only options that can be modified
#------------------------------------------------
def reload(config_args: Dict[str, Any]) -> bool:
    global _pipeline, _processType
    
    # Config
    config = DataObjects.PipelineConfig(**config_args)
    _processType = config.process_type
    _progress_tracker.Reset(total_models=5 if config.control_net_path is not None else 4)

    # Rebuild Pipeline
    _pipeline.unload_lora_weights()
    _pipeline = create_pipeline(config)

    # Load Lora
    Utils.load_lora_weights(_pipeline, config)

    # Memory
    Utils.configure_pipeline_memory(_pipeline, _execution_device, config)
    Utils.trim_memory(_isMemoryOffload)
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
# Get the log entires
#------------------------------------------------
def getLogs() -> list[str]:
    return Utils.get_output()


#------------------------------------------------
# Ge the last step latent
#------------------------------------------------
def getStepLatent() -> Buffer:
    return _step_latent


#------------------------------------------------
# Diffusers pipeline callback to caputer step artifacts
#------------------------------------------------
def _progress_callback(pipe, step: int, total_steps: int, info: Dict):
    global _step_latent
    if _cancel_event.is_set():
        pipe._interrupt = True
        raise Exception("Operation Canceled")

    latents = info.get("latents")
    if latents is not None:
        _step_latent = np.ascontiguousarray(latents.float().cpu())

    return info


#------------------------------------------------
# Generate Image/Video
#------------------------------------------------
def generate(
        inference_args: Dict[str, Any],
        input_tensors: Optional[List[Tuple[Sequence[float],Sequence[int]]]] = None,
        control_tensors: Optional[List[Tuple[Sequence[float],Sequence[int]]]] = None,
    ) -> Sequence[Buffer]:
    global _prompt_cache_key, _prompt_cache_value
    _cancel_event.clear()
    _pipeline._interrupt = False

    # Options
    options = DataObjects.PipelineOptions(**inference_args)

    #scheduler
    _pipeline.scheduler = Utils.create_scheduler(options.scheduler, options.scheduler_options, _pipeline.scheduler.config)

    #Lora Adapters
    Utils.set_lora_weights(_pipeline, options)

    # Input Images
    image = Utils.prepare_images(input_tensors)
    control_image = Utils.prepare_images(control_tensors)

    # Prompt Cache
    prompt_cache_key = (options.prompt, options.negative_prompt)
    if _prompt_cache_key != prompt_cache_key:
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

    if _processType in ("ImageToImage","ControlNetImageToImage"):
        pipeline_options.update({ "image": image, "strength": options.strength})

    if _processType == "ImageInpaint":
        pipeline_options.update({ "image": image[0], "mask_image": image[1], "strength": options.strength})

    if _processType == "ControlNetImage":
        pipeline_options.update({ "image": control_image })

    if _processType == "ControlNetImageToImage":
        pipeline_options.update({ "control_image": control_image })

    if _processType in ("ControlNetImage", "ControlNetImageToImage"):
        pipeline_options.update({
            "control_guidance_start": 0.0,
            "control_guidance_end": 1.0,
            "controlnet_conditioning_scale": options.control_net_scale
        })

    # Run Pipeline
    output = _pipeline(**pipeline_options)[0]

    # (Batch, Channel, Height, Width)
    output = output.transpose(0, 3, 1, 2).astype(np.float32)

    # Cleanup
    Utils.trim_memory(_isMemoryOffload)
    return [ np.ascontiguousarray(output) ]


#------------------------------------------------
# Create a new pipeline
#------------------------------------------------
def create_pipeline(config: DataObjects.PipelineConfig):
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
   
    # Build Pipeline
    _progress_tracker.Clear()
    pipeline = _pipelineMap[config.process_type]
    return pipeline.from_pretrained(
        config.base_model_path,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2, 
        unet=unet, 
        vae=vae, 
        torch_dtype=config.data_type,
        device_map=_device_map,
        local_files_only=True,
        **pipeline_kwargs
    )


#------------------------------------------------
# Load CLIPTextModel
#------------------------------------------------
def load_text_encoder(
        config: DataObjects.PipelineConfig, 
        pipeline_kwargs: Dict[str, str]
    ):

    if _pipeline and _pipeline.text_encoder:
        print(f"[Load] Loading cached TextEncoder")
        return _pipeline.text_encoder

    _progress_tracker.Initialize(0, "text_encoder")
    checkpoint = config.checkpoint_config.text_encoder_checkpoint
    if checkpoint:
        print(f"[Load] Loading checkpoint TextEncoder")
        text_encoder_checkpoint = Utils.load_component(
            StableDiffusionXLPipeline, 
            config.base_model_path, 
            checkpoint, 
            "text_encoder", 
            config.data_type,
            config.secure_token
        )
        if text_encoder_checkpoint:
            return text_encoder_checkpoint

    
    print(f"[Load] Loading TextEncoder")
    return CLIPTextModel.from_pretrained(
        config.base_model_path, 
        subfolder="text_encoder", 
        torch_dtype=config.data_type, 
        use_safetensors=True,
        **pipeline_kwargs
    )


#------------------------------------------------
# Load CLIPTextModelWithProjection 
#------------------------------------------------
def load_text_encoder_2(
        config: DataObjects.PipelineConfig, 
        pipeline_kwargs: Dict[str, str]
    ):

    if _pipeline and _pipeline.text_encoder_2:
        print(f"[Load] Loading cached TextEncoder2")
        return _pipeline.text_encoder_2
    
    _progress_tracker.Initialize(1, "text_encoder_2")
    checkpoint = config.checkpoint_config.text_encoder_checkpoint
    if checkpoint:
        print(f"[Load] Loading checkpoint TextEncoder2")
        text_encoder_checkpoint = Utils.load_component(
            StableDiffusionXLPipeline, 
            config.base_model_path, 
            checkpoint, 
            "text_encoder_2", 
            config.data_type,
            config.secure_token
        )
        if text_encoder_checkpoint:
            return text_encoder_checkpoint

    print(f"[Load] Loading TextEncoder2")
    return CLIPTextModelWithProjection.from_pretrained(
        config.base_model_path, 
        subfolder="text_encoder_2",
        torch_dtype=config.data_type, 
        quantization_config=_quant_config_transformers, 
        use_safetensors=True,
        **pipeline_kwargs
    )


#------------------------------------------------
# Load UNet2DConditionModel
#------------------------------------------------
def load_unet(
        config: DataObjects.PipelineConfig, 
        pipeline_kwargs: Dict[str, str]
    ):

    if _pipeline and _pipeline.unet:
        print(f"[Load] Loading cached Unet")
        return _pipeline.unet
    
    _progress_tracker.Initialize(2, "unet")
    checkpoint= config.checkpoint_config.model_checkpoint
    if checkpoint:
        print(f"[Load] Loading checkpoint Unet")
        unet_checkpoint = UNet2DConditionModel.from_single_file(
            checkpoint, 
            config=_pipeline_config["unet"],
            torch_dtype=config.data_type, 
            use_safetensors=True, 
            local_files_only=False,
            token=config.secure_token,
            quantization_config=Quantization.get_single_file_config(config)
        )
        Quantization.quantize_model(config, unet_checkpoint)
        return unet_checkpoint
    
    print(f"[Load] Loading Unet")
    return UNet2DConditionModel.from_pretrained(
        config.base_model_path, 
        subfolder="unet", 
        torch_dtype=config.data_type, 
        quantization_config=_quant_config_diffusers, 
        use_safetensors=True,
        **pipeline_kwargs
    )


#------------------------------------------------
# Load AutoencoderKL
#------------------------------------------------
def load_vae(
        config: DataObjects.PipelineConfig, 
        pipeline_kwargs: Dict[str, str]
    ):

    if _pipeline and _pipeline.vae:
        print(f"[Load] Loading cached Vae")
        return _pipeline.vae

    _progress_tracker.Initialize(3, "vae")
    checkpoint = config.checkpoint_config.vae_checkpoint
    if checkpoint:
        print(f"[Load] Loading checkpoint Vae")
        vae_checkpoint = Utils.load_component(
            StableDiffusionXLPipeline, 
            config.base_model_path, 
            checkpoint, 
            "vae", 
            config.data_type,
            config.secure_token
        )
        if vae_checkpoint:
            return vae_checkpoint
    
    print(f"[Load] Loading Vae")
    return AutoencoderKL.from_pretrained(
        config.base_model_path, 
        subfolder="vae", 
        torch_dtype=config.data_type, 
        use_safetensors=True,
        **pipeline_kwargs
    )


#------------------------------------------------
# Load ControlNetModel
#------------------------------------------------
def load_control_net(
        config: DataObjects.PipelineConfig, 
        pipeline_kwargs: Dict[str, str]
    ):
    global _control_net_path, _control_net_cache

    if _control_net_cache and _control_net_path == config.control_net_path:
        print(f"[Load] Loading cached ControlNet")
        return _control_net_cache

    if config.control_net_path is None:
        _control_net_path = None
        _control_net_cache = None
        return None
    
    print(f"[Load] Loading ControlNet")
    _control_net_path = config.control_net_path
    _progress_tracker.Initialize(4, "control_net")
    _control_net_cache = ControlNetModel.from_pretrained(
        _control_net_path, 
        torch_dtype=config.data_type,
        use_safetensors=True,
    )
    return _control_net_cache
       

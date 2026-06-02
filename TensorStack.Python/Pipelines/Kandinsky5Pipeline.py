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
from transformers import CLIPTokenizer, CLIPTextModel, Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration
from diffusers import (
    AutoencoderKL,
    AutoencoderKLHunyuanVideo,
    Kandinsky5Transformer3DModel,
    Kandinsky5T2IPipeline,
    Kandinsky5I2IPipeline,
    Kandinsky5T2VPipeline,
    Kandinsky5I2VPipeline
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
_is_video_pipeline = False
_cancel_event = Event()
_stopwatch = None
_pipelineMap = {
    ProcessType.TextToImage: Kandinsky5T2IPipeline,
    ProcessType.ImageEdit: Kandinsky5I2IPipeline,
    ProcessType.TextToVideo: Kandinsky5T2VPipeline,
    ProcessType.ImageToVideo: Kandinsky5I2VPipeline
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

    steps = pipe._num_timesteps
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
# Load Qwen2Tokenizer
#------------------------------------------------
def load_tokenizer(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    if _pipeline and _pipeline.tokenizer:
        print(f"[Load] Loading Cached Tokenizer")
        return _pipeline.tokenizer

    tokenizer_path: Path = _model_config["tokenizer"]
    tokenizer_config: Path = _model_config["tokenizer_config"]

    # 1. Load from pretrained folder
    print(f"[Load] Loading Pretrained Tokenizer")
    tokenizer = Qwen2Tokenizer.from_pretrained(
        tokenizer_path,
        config=tokenizer_config,
        dtype=config.data_type,
        **pipeline_kwargs
    )
    return tokenizer


#------------------------------------------------
# Load CLIPTokenizer
#------------------------------------------------
def load_tokenizer_2(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    if _pipeline and _pipeline.tokenizer_2:
        print(f"[Load] Loading Cached Tokenizer2")
        return _pipeline.tokenizer_2

    tokenizer_2_path: Path = _model_config["tokenizer_2"]
    tokenizer_2_config: Path = _model_config["tokenizer_2_config"]

    # 1. Load from pretrained folder
    print(f"[Load] Loading Pretrained Tokenizer2")
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        tokenizer_2_path,
        config=tokenizer_2_config,
        dtype=config.data_type,
        **pipeline_kwargs
    )
    return tokenizer_2


#------------------------------------------------
# Load Qwen2_5_VLForConditionalGeneration
#------------------------------------------------
def load_text_encoder(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    if _pipeline and _pipeline.text_encoder:
        print(f"[Load] Loading Cached TextEncoder")
        return _pipeline.text_encoder

    text_encoder_path: Path = _model_config["text_encoder"]
    text_encoder_config: Path = _model_config["text_encoder_config"]

    # 1. Load from pretrained folder
    print(f"[Load] Loading Pretrained TextEncoder")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
# Load CLIPTextModel
#------------------------------------------------
def load_text_encoder_2(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    if _pipeline and _pipeline.text_encoder_2:
        print(f"[Load] Loading Cached TextEncoder2")
        return _pipeline.text_encoder_2

    text_encoder_2_path: Path = _model_config["text_encoder_2"]
    text_encoder_2_config: Path = _model_config["text_encoder_2_config"]
    single_path: Path = _model_config["single_file"]
    template_path: Path  = _model_config["template"]

    # 2. Load component from single file
    if single_path and single_path.is_file():
        print(f"[Load] Loading Component TextEncoder2")
        text_encoder_2 = Utils.from_component(Kandinsky5T2IPipeline, "text_encoder_2", single_path, template_path, _device_map, config.data_type)
        if text_encoder_2:
            Utils.trim_memory(True)
            return text_encoder_2

    # 1. Load from pretrained folder
    print(f"[Load] Loading Pretrained TextEncoder2")
    text_encoder_2 = CLIPTextModel.from_pretrained(
        text_encoder_2_path,
        config=text_encoder_2_config,
        dtype=config.data_type,
        device_map=_device_map,
        quantization_config=Quantization.auto_pretrained_config(config, QuantTarget.TEXT_ENCODER),
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return text_encoder_2


#------------------------------------------------
# Load Kandinsky5Transformer3DModel
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
        transformer =  Kandinsky5Transformer3DModel.from_single_file(
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
    transformer =  Kandinsky5Transformer3DModel.from_pretrained(
        str(transformer_path),
        torch_dtype=config.data_type,
        device_map=_device_map,
        quantization_config=Quantization.auto_pretrained_config(config, QuantTarget.TRANSFORMER),
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return transformer


#------------------------------------------------
# Load AutoencoderKL
#------------------------------------------------
def load_vae_image(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
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
        auto_encoder =  AutoencoderKL.from_single_file(
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
        auto_encoder = Utils.from_component(Kandinsky5T2IPipeline, "vae", single_path, template_path, _device_map, config.data_type)
        if auto_encoder:
            Utils.trim_memory(True)
            return auto_encoder

    # 3. Load from pretrained folder
    print(f"[Load] Loading Pretrained Vae")
    auto_encoder = AutoencoderKL.from_pretrained(
        str(vae_path),
        torch_dtype=config.data_type,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return auto_encoder


#------------------------------------------------
# Load AutoencoderKLHunyuanVideo
#------------------------------------------------
def load_vae_video(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
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
        auto_encoder =  AutoencoderKLHunyuanVideo.from_single_file(
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
        auto_encoder = Utils.from_component(Kandinsky5T2VPipeline, "vae", single_path, template_path, _device_map, config.data_type)
        if auto_encoder:
            Utils.trim_memory(True)
            return auto_encoder

    # 3. Load from pretrained folder
    print(f"[Load] Loading Pretrained Vae")
    auto_encoder = AutoencoderKLHunyuanVideo.from_pretrained(
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

    # print(f"[Load] Loading Pretrained ControlNet, IsOffline: {config.control_net.is_offline_mode}")
    # _control_net_name = config.control_net.name
    # _progress_tracker.Initialize(4, "control_net")
    # _control_net_cache = ControlNetModel.from_pretrained(
    #     config.control_net.path,
    #     torch_dtype=config.data_type,
    #     device_map=_device_map,
    #     **pipeline_kwargs
    # )
    return None


#------------------------------------------------
# Create a new pipeline
#------------------------------------------------
def create_pipeline(config: DataObjects.PipelineConfig):
    global _is_video_pipeline
    template_path: Path = _model_config["template"]
    pipeline_kwargs = {
        "variant": config.variant,
        "use_safetensors":True,
        "low_cpu_mem_usage":True,
        "local_files_only":True,
    }

    if _processType in (ProcessType.TextToVideo, ProcessType.ImageToVideo):
        _is_video_pipeline = True

    # Load Models
    tokenizer = load_tokenizer(config, pipeline_kwargs)
    tokenizer_2 = load_tokenizer_2(config, pipeline_kwargs)
    text_encoder = load_text_encoder(config, pipeline_kwargs)
    text_encoder_2 = load_text_encoder_2(config, pipeline_kwargs)
    transformer = load_transformer(config, pipeline_kwargs)
    vae = (
        load_vae_video(config, pipeline_kwargs)
        if _is_video_pipeline
        else load_vae_image(config, pipeline_kwargs)
    )
    control_net = load_control_net(config, pipeline_kwargs)
    if control_net is not None:
        pipeline_kwargs.update({"controlnet": control_net})

    # Build Pipeline
    pipeline = _pipelineMap[_processType]
    return pipeline.from_pretrained(
        template_path,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        transformer=transformer,
        vae=vae,
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
    prompt_cache_key = (options.prompt, options.negative_prompt, options.guidance_scale > 1)
    if _prompt_cache_key != prompt_cache_key:
        print(f"[Generate] Encoding prompt")
        with torch.no_grad():
            negative_prompt_value = (None, None, None)
            prompt_value = _pipeline.encode_prompt(
                prompt=options.prompt,
                device=_pipeline._execution_device,
                max_sequence_length=512
            )
            if options.guidance_scale > 1:
                negative_prompt_value = _pipeline.encode_prompt(
                    prompt=options.negative_prompt,
                    device=_pipeline._execution_device,
                    max_sequence_length=512
                )

            _prompt_cache_value = (prompt_value, negative_prompt_value)
            _prompt_cache_key = prompt_cache_key

    # Notify
    Utils.notification_push(key="Generate", subkey="Encode", elapsed=_stopwatch.reset())

    # Pipeline Options
    (prompt_cache, negative_prompt_cache) = _prompt_cache_value
    (prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens) = prompt_cache
    (negative_prompt_embeds_qwen, negative_prompt_embeds_clip, negative_prompt_cu_seqlens) = negative_prompt_cache
    pipeline_options = {
        "prompt_embeds_qwen": prompt_embeds_qwen,
        "prompt_embeds_clip": prompt_embeds_clip,
        "prompt_cu_seqlens": prompt_cu_seqlens,
        "negative_prompt_embeds_qwen": negative_prompt_embeds_qwen,
        "negative_prompt_embeds_clip": negative_prompt_embeds_clip,
        "negative_prompt_cu_seqlens": negative_prompt_cu_seqlens,
        "height": options.height,
        "width": options.width,
        "generator": _generator.manual_seed(options.seed),
        "guidance_scale": options.guidance_scale,
        "num_inference_steps": options.steps,
        "output_type": "np",
        "callback_on_step_end": _progress_callback,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }

    if _processType in (ProcessType.ImageEdit, ProcessType.ImageToVideo):
        pipeline_options.update({ "image": images})
    if _processType in (ProcessType.TextToVideo, ProcessType.ImageToVideo):
        pipeline_options.update({ "num_frames": options.frames })

    # Run Pipeline
    output = _pipeline(**pipeline_options)[0]

    if _is_video_pipeline == True:
        # (Frames, Channel, Height, Width)
        output = output.transpose(0, 1, 4, 2, 3).squeeze(axis=0).astype(np.float32)

    if _is_video_pipeline == False:
        # (Batch, Channel, Height, Width)
        output = output.transpose(0, 3, 1, 2).astype(np.float32)

    # Notify
    Utils.notification_push(key="Generate", subkey="Decode", elapsed = _stopwatch.reset())
    Utils.notification_push(key="Generate", subkey="Complete", elapsed = _stopwatch.stop())

    # Cleanup
    Utils.trim_memory(_isMemoryOffload)
    return [ np.ascontiguousarray(output) ]
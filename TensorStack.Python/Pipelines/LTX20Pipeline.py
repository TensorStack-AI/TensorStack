import tensorstack.utils as Utils
import tensorstack.export as Export
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
from transformers import GemmaTokenizer, Gemma3ForConditionalGeneration
from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    LTX2VideoTransformer3DModel,
    LTX2Pipeline,
    LTX2ConditionPipeline
)
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder, LTX2VocoderWithBWE
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, T2V_DEFAULT_SYSTEM_PROMPT, I2V_DEFAULT_SYSTEM_PROMPT

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
    ProcessType.TextToVideo: LTX2Pipeline,
    ProcessType.ImageToVideo: LTX2ConditionPipeline
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
# Load GemmaTokenizer
#------------------------------------------------
def load_tokenizer(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    if _pipeline and _pipeline.tokenizer:
        print(f"[Load] Loading Cached Tokenizer")
        return _pipeline.tokenizer

    tokenizer_path: Path = _model_config["tokenizer"]
    tokenizer_config: Path = _model_config["tokenizer_config"]

    # 1. Load from pretrained folder
    print(f"[Load] Loading Pretrained Tokenizer")
    tokenizer = GemmaTokenizer.from_pretrained(
        tokenizer_path,
        config=tokenizer_config,
        dtype=config.data_type,
        fix_mistral_regex=True,
        **pipeline_kwargs
    )
    return tokenizer


#------------------------------------------------
# Load Gemma3ForConditionalGeneration
#------------------------------------------------
def load_text_encoder(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    if _pipeline and _pipeline.text_encoder:
        print(f"[Load] Loading Cached TextEncoder")
        return _pipeline.text_encoder

    text_encoder_path: Path = _model_config["text_encoder"]
    text_encoder_config: Path = _model_config["text_encoder_config"]

    # 1. Load from pretrained folder
    print(f"[Load] Loading Pretrained TextEncoder")
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
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
# Load LTX2VideoTransformer3DModel
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
        transformer =  LTX2VideoTransformer3DModel.from_single_file(
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
    transformer =  LTX2VideoTransformer3DModel.from_pretrained(
        str(transformer_path),
        torch_dtype=config.data_type,
        device_map=None,
        quantization_config=Quantization.auto_pretrained_config(config, QuantTarget.TRANSFORMER),
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

    vae_path: Path = _model_config["vae"]
    vae_config: Path = _model_config["vae_config"]
    single_path: Path = _model_config["single_file"]
    template_path: Path  = _model_config["template"]

    # 1. Load from single file
    if vae_path.is_file():
        print(f"[Load] Loading SingleFile Vae")
        auto_encoder =  AutoencoderKLLTX2Video.from_single_file(
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
        auto_encoder = Utils.from_component(LTX2Pipeline, "vae", single_path, template_path, _device_map, config.data_type)
        if auto_encoder:
            Utils.trim_memory(True)
            return auto_encoder

    # 3. Load from pretrained folder
    print(f"[Load] Loading Pretrained Vae")
    auto_encoder = AutoencoderKLLTX2Video.from_pretrained(
        str(vae_path),
        torch_dtype=config.data_type,
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
        print(f"[Load] Loading Cached AudioVae")
        return _pipeline.audio_vae

    audio_vae_path: Path = _model_config["audio_vae"]
    audio_vae_config: Path = _model_config["audio_vae_config"]
    single_path: Path = _model_config["single_file"]
    template_path: Path  = _model_config["template"]

    # 1. Load from single file
    if audio_vae_path.is_file():
        print(f"[Load] Loading SingleFile AudioVae")
        audio_vae =  AutoencoderKLLTX2Audio.from_single_file(
            str(audio_vae_path),
            config=str(audio_vae_config),
            torch_dtype=config.data_type,
            device_map=_device_map,
            **pipeline_kwargs
        )
        Utils.trim_memory(True)
        return audio_vae

    # 2. Load component from single file
    if single_path and single_path.is_file():
        print(f"[Load] Loading Component AudioVae")
        audio_vae = Utils.from_component(LTX2Pipeline, "audio_vae", single_path, template_path, _device_map, config.data_type)
        if audio_vae:
            Utils.trim_memory(True)
            return audio_vae

    # 3. Load from pretrained folder
    print(f"[Load] Loading Pretrained AudioVae")
    audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
        str(audio_vae_path),
        torch_dtype=config.data_type,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return audio_vae


#------------------------------------------------
# Load Vocoder
#------------------------------------------------
def load_vocoder(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    if _pipeline and _pipeline.vocoder:
        print(f"[Load] Loading Cached Vocoder")
        return _pipeline.vocoder

    vocoder_path: Path = _model_config["vocoder"]
    vocoder_config: Path = _model_config["vocoder_config"]
    single_path: Path = _model_config["single_file"]
    template_path: Path  = _model_config["template"]

    # 1. Load from single file
    if vocoder_path.is_file():
        print(f"[Load] Loading SingleFile Vocoder")
        vocoder =  LTX2Vocoder.from_single_file(
            str(vocoder_path),
            config=str(vocoder_config),
            torch_dtype=config.data_type,
            device_map=_device_map,
            **pipeline_kwargs
        )
        Utils.trim_memory(True)
        return vocoder

    # 2. Load component from single file
    if single_path and single_path.is_file():
        print(f"[Load] Loading Component Vocoder")
        vocoder = Utils.from_component(LTX2Pipeline, "vocoder", single_path, template_path, _device_map, config.data_type)
        if vocoder:
            Utils.trim_memory(True)
            return vocoder

    # 3. Load from pretrained folder
    print(f"[Load] Loading Pretrained Vocoder")
    vocoder = LTX2Vocoder.from_pretrained(
        str(vocoder_path),
        torch_dtype=config.data_type,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return vocoder


#------------------------------------------------
# Load VocoderWithBWE
#------------------------------------------------
def load_vocoder_bwe(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    if _pipeline and _pipeline.vocoder:
        print(f"[Load] Loading Cached VocoderWithBWE")
        return _pipeline.vocoder

    vocoder_path: Path = _model_config["vocoder"]
    vocoder_config: Path = _model_config["vocoder_config"]
    single_path: Path = _model_config["single_file"]
    template_path: Path  = _model_config["template"]

    # 1. Load from single file
    if vocoder_path.is_file():
        print(f"[Load] Loading SingleFile VocoderWithBWE")
        vocoder =  LTX2VocoderWithBWE.from_single_file(
            str(vocoder_path),
            config=str(vocoder_config),
            torch_dtype=config.data_type,
            device_map=_device_map,
            **pipeline_kwargs
        )
        Utils.trim_memory(True)
        return vocoder

    # 2. Load component from single file
    if single_path and single_path.is_file():
        print(f"[Load] Loading Component VocoderWithBWE")
        vocoder = Utils.from_component(LTX2Pipeline, "vocoder", single_path, template_path, _device_map, config.data_type)
        if vocoder:
            Utils.trim_memory(True)
            return vocoder

    # 3. Load from pretrained folder
    print(f"[Load] Loading Pretrained VocoderWithBWE")
    vocoder = LTX2VocoderWithBWE.from_pretrained(
        str(vocoder_path),
        torch_dtype=config.data_type,
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

    connectors_path: Path = _model_config["connectors"]
    connectors_config: Path = _model_config["connectors_config"]
    single_path: Path = _model_config["single_file"]
    template_path: Path  = _model_config["template"]

    # 1. Load from single file
    if connectors_path.is_file():
        print(f"[Load] Loading SingleFile AudioVae")
        connectors =  LTX2TextConnectors.from_single_file(
            str(connectors_path),
            config=str(connectors_config),
            torch_dtype=config.data_type,
            device_map=_device_map,
            **pipeline_kwargs
        )
        Utils.trim_memory(True)
        return connectors

    # 2. Load component from single file
    if single_path and single_path.is_file():
        print(f"[Load] Loading Component AudioVae")
        connectors = Utils.from_component(LTX2Pipeline, "connectors", single_path, template_path, _device_map, config.data_type)
        if connectors:
            Utils.trim_memory(True)
            return connectors

    # 3. Load from pretrained folder
    print(f"[Load] Loading Pretrained AudioVae")
    connectors = LTX2TextConnectors.from_pretrained(
        str(connectors_path),
        torch_dtype=config.data_type,
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

    # print(f"[Load] Loading Pretrained ControlNet")
    # _control_net_name = config.control_net.name
    # _progress_tracker.Initialize(3, "control_net")
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
    vae = load_vae_video(config, pipeline_kwargs)
    audio_vae = load_vae_audio(config, pipeline_kwargs)
    connectors = load_connectors(config, pipeline_kwargs)
    control_net = load_control_net(config, pipeline_kwargs)
    vocoder = (
        load_vocoder(config, pipeline_kwargs)
        if config.template == "LTX_20"
        else load_vocoder_bwe(config, pipeline_kwargs)
    )
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
        audio_vae=audio_vae,
        vocoder=vocoder,
        connectors=connectors,
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

    #Lora Adapters
    Utils.set_lora_weights(_pipeline, options)

    # Notify
    Utils.notification_push(key="Generate", subkey="Initialize", elapsed=_stopwatch.reset())

    # Prompt Cache
    negative_prompt = options.negative_prompt if options.negative_prompt else DEFAULT_NEGATIVE_PROMPT
    prompt_cache_key = (options.prompt, negative_prompt, options.guidance_scale > 1.0)
    if _prompt_cache_key != prompt_cache_key:
        print(f"[Generate] Encoding prompt")
        with torch.no_grad():
            _prompt_cache_value = _pipeline.encode_prompt(
                prompt=options.prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=options.guidance_scale > 1.0,
                num_videos_per_prompt=1
            )
            _prompt_cache_key = prompt_cache_key
            Utils.trim_memory(_isMemoryOffload)

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

    # Video Conditions
    if _processType == ProcessType.ImageToVideo:
        conditions = None
        if image_count == 1:
            conditions = LTX2VideoCondition(frames=images, index=0, strength=1.0)
        elif image_count == 2:
            first_frame = LTX2VideoCondition(frames=images[0], index=0, strength=1.0)
            last_frame = LTX2VideoCondition(frames=images[1], index=-1, strength=1.0)
            conditions = [first_frame, last_frame]

        pipeline_options.update({ "conditions": conditions })

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

import tensorstack.utils as Utils
import tensorstack.data_objects as DataObjects
import tensorstack.quantization as Quantization
from tensorstack.enums import ProcessType, QuantTarget
Utils.redirect_output()
Utils.create_services()

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from threading import Event
from collections.abc import Buffer
from typing import Dict, Sequence, List, Tuple, Optional, Any
from transformers import Qwen2Tokenizer, AutoModel
from diffusers import (
    AutoencoderOobleck,
    AceStepTransformer1DModel,
    AceStepConditionEncoder,
    AceStepAudioTokenizer,
    AceStepAudioTokenDetokenizer,
    AceStepPipeline
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
    ProcessType.TextToAudio: AceStepPipeline,
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
# Load Tokenizer Qwen2Tokenizer
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
# Load TextEncoder AutoModel
#------------------------------------------------
def load_text_encoder(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    if _pipeline and _pipeline.text_encoder:
        print(f"[Load] Loading Cached TextEncoder")
        return _pipeline.text_encoder

    text_encoder_path: Path = _model_config["text_encoder"]
    text_encoder_config: Path = _model_config["text_encoder_config"]

    # 1. Load from pretrained folder
    print(f"[Load] Loading Pretrained TextEncoder")
    text_encoder = AutoModel.from_pretrained(
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
# Load AceStepTransformer1DModel
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
        transformer =  AceStepTransformer1DModel.from_single_file(
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
    transformer =  AceStepTransformer1DModel.from_pretrained(
        str(transformer_path),
        torch_dtype=config.data_type,
        device_map=_device_map,
        quantization_config=Quantization.auto_pretrained_config(config, QuantTarget.TRANSFORMER),
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

    vae_path: Path = _model_config["vae"]
    vae_config: Path = _model_config["vae_config"]
    single_path: Path = _model_config["single_file"]
    template_path: Path  = _model_config["template"]

    # 1. Load from single file
    if vae_path.is_file():
        print(f"[Load] Loading SingleFile Vae")
        auto_encoder =  AutoencoderOobleck.from_single_file(
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
        auto_encoder = Utils.from_component(AceStepPipeline, "vae", single_path, template_path, _device_map, config.data_type)
        if auto_encoder:
            Utils.trim_memory(True)
            return auto_encoder

    # 3. Load from pretrained folder
    print(f"[Load] Loading Pretrained Vae")
    auto_encoder = AutoencoderOobleck.from_pretrained(
        str(vae_path),
        torch_dtype=config.data_type,
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

    condition_encoder_path: Path = _model_config["condition_encoder"]
    condition_encoder_config: Path = _model_config["condition_encoder_config"]
    single_path: Path = _model_config["single_file"]
    template_path: Path  = _model_config["template"]

    # 1. Load from single file
    if condition_encoder_path.is_file():
        print(f"[Load] Loading SingleFile ConditionEncoder")
        condition_encoder =  AceStepConditionEncoder.from_single_file(
            str(condition_encoder_path),
            config=str(condition_encoder_config),
            torch_dtype=config.data_type,
            device_map=_device_map,
            **pipeline_kwargs
        )
        Utils.trim_memory(True)
        return condition_encoder

    # 2. Load component from single file
    if single_path and single_path.is_file():
        print(f"[Load] Loading Component ConditionEncoder")
        condition_encoder = Utils.from_component(AceStepPipeline, "condition_encoder", single_path, template_path, _device_map, config.data_type)
        if condition_encoder:
            Utils.trim_memory(True)
            return condition_encoder

    # 3. Load from pretrained folder
    print(f"[Load] Loading Pretrained ConditionEncoder")
    condition_encoder = AceStepConditionEncoder.from_pretrained(
        str(condition_encoder_path),
        torch_dtype=config.data_type,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return condition_encoder


#------------------------------------------------
# Load AceStepAudioTokenizer
#------------------------------------------------
def load_audio_tokenizer(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    if _pipeline and _pipeline.audio_tokenizer:
        print(f"[Load] Loading Cached AudioTokenizer")
        return _pipeline.audio_tokenizer

    audio_tokenizer_path: Path = _model_config["audio_tokenizer"]
    audio_tokenizer_config: Path = _model_config["audio_tokenizer_config"]
    single_path: Path = _model_config["single_file"]
    template_path: Path  = _model_config["template"]
    if not audio_tokenizer_path.exists():
        print(f"[Load] AudioTokenizer not found!")
        return None

    # 1. Load from single file
    if audio_tokenizer_path.is_file():
        print(f"[Load] Loading SingleFile AudioTokenizer")
        audio_tokenizer =  AceStepAudioTokenizer.from_single_file(
            str(audio_tokenizer_path),
            config=str(audio_tokenizer_config),
            torch_dtype=config.data_type,
            device_map=_device_map,
            **pipeline_kwargs
        )
        Utils.trim_memory(True)
        return audio_tokenizer

    # 2. Load component from single file
    if single_path and single_path.is_file():
        print(f"[Load] Loading Component AudioTokenizer")
        audio_tokenizer = Utils.from_component(AceStepPipeline, "audio_tokenizer", single_path, template_path, _device_map, config.data_type)
        if audio_tokenizer:
            Utils.trim_memory(True)
            return audio_tokenizer

    # 3. Load from pretrained folder
    print(f"[Load] Loading Pretrained AudioTokenizer")
    audio_tokenizer = AceStepAudioTokenizer.from_pretrained(
        str(audio_tokenizer_path),
        torch_dtype=config.data_type,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return audio_tokenizer



#------------------------------------------------
# Load AceStepAudioTokenDetokenizer
#------------------------------------------------
def load_audio_token_detokenizer(config: DataObjects.PipelineConfig, pipeline_kwargs: Dict[str, str]):
    if _pipeline and _pipeline.audio_token_detokenizer:
        print(f"[Load] Loading Cached AudioTokenDetokenizer")
        return _pipeline.audio_token_detokenizer

    audio_token_detokenizer_path: Path = _model_config["audio_token_detokenizer"]
    audio_token_detokenizer_config: Path = _model_config["audio_token_detokenizer_config"]
    single_path: Path = _model_config["single_file"]
    template_path: Path  = _model_config["template"]
    if not audio_token_detokenizer_path.exists():
        print(f"[Load] AudioTokenDetokenizer not found!")
        return None

    # 1. Load from single file
    if audio_token_detokenizer_path.is_file():
        print(f"[Load] Loading SingleFile AudioTokenDetokenizer")
        audio_token_detokenizer =  AceStepAudioTokenDetokenizer.from_single_file(
            str(audio_token_detokenizer_path),
            config=str(audio_token_detokenizer_config),
            torch_dtype=config.data_type,
            device_map=_device_map,
            **pipeline_kwargs
        )
        Utils.trim_memory(True)
        return audio_token_detokenizer

    # 2. Load component from single file
    if single_path and single_path.is_file():
        print(f"[Load] Loading Component AudioTokenDetokenizer")
        audio_token_detokenizer = Utils.from_component(AceStepPipeline, "audio_token_detokenizer", single_path, template_path, _device_map, config.data_type)
        if audio_token_detokenizer:
            Utils.trim_memory(True)
            return audio_token_detokenizer

    # 3. Load from pretrained folder
    print(f"[Load] Loading Pretrained AudioTokenDetokenizer")
    audio_token_detokenizer = AceStepAudioTokenDetokenizer.from_pretrained(
        str(audio_token_detokenizer_path),
        torch_dtype=config.data_type,
        device_map=_device_map,
        **pipeline_kwargs
    )
    Utils.trim_memory(True)
    return audio_token_detokenizer


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
    vae = load_vae(config, pipeline_kwargs)
    condition_encoder = load_condition_encoder(config, pipeline_kwargs)
    audio_tokenizer = load_audio_tokenizer(config, pipeline_kwargs)
    audio_token_detokenizer = load_audio_token_detokenizer(config, pipeline_kwargs)

    # Build Pipeline
    pipeline = _pipelineMap[_processType]
    return pipeline.from_pretrained(
        template_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        condition_encoder=condition_encoder,
        audio_tokenizer=audio_tokenizer,
        audio_token_detokenizer=audio_token_detokenizer,
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
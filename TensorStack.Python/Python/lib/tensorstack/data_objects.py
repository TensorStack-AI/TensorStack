
from dataclasses import dataclass, fields
from typing import Optional, Union, Sequence, get_args, get_origin
from tensorstack.enums import QuantType
import torch

def get_data_type(dtype: str):
    if dtype == "float8_e5m2":
        return torch.float8_e5m2
    if dtype == "float8_e4m3fn":
        return torch.float8_e4m3fn
    if dtype == "float8":
        return torch.float8_e4m3fn
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "int8":
        return torch.int8
    if dtype == "int16":
        return torch.int16
    if dtype == "int32":
        return torch.int32
    if dtype == "int64":
        return torch.int64
    if dtype == "float32":
        return torch.float32
    if dtype == "int4":
        return torch.int
    return torch.float


@dataclass(slots=True)
class CheckpointConfig:
    model_checkpoint: Optional[str] = None
    vae_checkpoint: Optional[str] = None
    text_encoder_checkpoint: Optional[str] = None


@dataclass(slots=True)
class LoraConfig:
    path: str
    name: str
    weights: str


@dataclass(slots=True)
class ControlNetConfig:
    path: Optional[str] = None
    name: Optional[str] = None


@dataclass(slots=True)
class LoraOption:
    name: str
    strength: float

    def __post_init__(self):
        self.strength = float(self.strength)




@dataclass(slots=True)
class PipelineConfig:
    # Required / core
    base_model_path: str
    pipeline: str
    process_type: str
    memory_mode: str
    is_gguf: bool = False 

    # Device
    device: str = "cuda"
    device_id: int = 0
    device_bus_id: int = 0

    data_type: Union[str, torch.dtype] = "bfloat16"
    quant_type: QuantType = QuantType.Q16Bit
    
    is_optimize_device_enabled: bool = False
    is_optimize_channels_enabled: bool = False

    # HF / loading
    variant: Optional[str] = None
    cache_directory: Optional[str] = None
    secure_token: Optional[str] = None

    lora_adapters: Optional[Sequence[LoraConfig]] = None
    control_net: Optional[ControlNetConfig] = None
    checkpoint_config: Optional[CheckpointConfig] = None
  
    def __post_init__(self):
        self.data_type = get_data_type(self.data_type)
        if (self.lora_adapters is not None and isinstance(self.lora_adapters, Sequence)):
            self.lora_adapters = [LoraConfig(**dict(cfg)) for cfg in self.lora_adapters or []]
        if (self.checkpoint_config is not None and isinstance(self.checkpoint_config, dict)):
            self.checkpoint_config = CheckpointConfig(**self.checkpoint_config)
        elif self.checkpoint_config is None:
            self.checkpoint_config = CheckpointConfig()
        if (self.control_net is not None and isinstance(self.control_net, dict)):
            self.control_net = ControlNetConfig(**self.control_net)
        elif self.control_net is None:
            self.control_net = ControlNetConfig()

        if isinstance(self.quant_type, str):
            try:
                self.quant_type = QuantType[self.quant_type]
            except KeyError:
                self.quant_type = QuantType.Q16Bit


        model_ckpt = getattr(self.checkpoint_config, "model_checkpoint", None)
        self.is_gguf = (
            (model_ckpt is not None and str(model_ckpt).lower().endswith(".gguf")) or
            (getattr(self, "base_model_path", None) is not None and str(self.base_model_path).lower().endswith(".gguf"))
        )


@dataclass(slots=True)
class SchedulerOptions:
    Scheduler: str
    num_train_timesteps: Optional[int] = None 
    original_inference_steps: Optional[int] = None 
    base_image_seq_len: Optional[int] = None 
    max_image_seq_len: Optional[int ] = None 

    beta_schedule: Optional[str] = None             # BetaScheduleType
    beta_start: Optional[float] = None 
    beta_end: Optional[float] = None 
  
    prediction_type: Optional[str] = None           # PredictionType
    timestep_spacing: Optional[str] = None          # TimestepSpacingType
    steps_offset: Optional[int] = None 

    clip_sample: Optional[bool] = None 
    clip_sample_range: Optional[float] = None 
    sample_max_value: Optional[float] = None 

    thresholding: Optional[bool] = None 
    dynamic_thresholding_ratio: Optional[float] = None 
    variance_type: Optional[str] = None             # VarianceType
  
    use_karras_sigmas: Optional[bool] = None 
    use_beta_sigmas: Optional[bool] = None 
    use_exponential_sigmas: Optional[bool] = None 
    use_flow_sigmas: Optional[bool ] = None 

    sigma_min: Optional[float] = None
    sigma_max: Optional[float] = None
    final_sigmas_type: Optional[str] = None         # FinalSigmasType

    interpolation_type: Optional[str] = None        # InterpolationType
    timestep_type: Optional[str] = None             # TimestepType
    rescale_betas_zero_snr: Optional[bool] = None 
    set_alpha_to_one: Optional[bool] = None 
    timestep_scaling: Optional[float] = None 

    shift: Optional[float] = None 
    base_shift: Optional[float] = None 
    max_shift: Optional[float] = None 
    shift_terminal: Optional[float] = None
    use_dynamic_shifting: Optional[bool] = None 
    flow_shift: Optional[float] = None 
    snr_shift_scale: Optional[float] = None 

    time_shift_type: Optional[str] = None           # TimeShiftType
    rho: Optional[float] = None 

    solver_order: Optional[int] = None 
    solver_type: Optional[str] = None               # SolverType
    algorithm_type: Optional[str] = None            # AlgorithmType
    lower_order_final: Optional[bool] = None 

    stochastic_sampling: Optional[bool] = None 
    eta: Optional[float] = None 
    s_noise: Optional[float] = None 

    invert_sigmas: Optional[bool] = None 
    skip_prk_steps: Optional[bool] = None 
    predict_x0: Optional[bool] = None 
    euler_at_final: Optional[bool] = None 

    use_lu_lambdas: Optional[bool] = None 
    noise_sampler_seed: Optional[int] = None
    sigma_data: Optional[float] = None 
    sigma_schedule: Optional[str] = None            # SigmaScheduleType
    upscale_mode: Optional[str] = None              # UpscaleModeType

    stages: Optional[int] = None 
    gamma: Optional[float] = None 
    predictor_order: Optional[int] = None 
    corrector_order: Optional[int] = None 

    scale_factors: Optional[Sequence[float]] = None 
    stage_range: Optional[Sequence[float]] = None 
    disable_corrector: Optional[Sequence[int]] = None 

    s: Optional[float] = None 
    scaler: Optional[float] = None 

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            origin = get_origin(field.type)
            args = get_args(field.type)
            if field.type is float or (origin is Union and float in args):
                setattr(self, field.name, float(value))
            elif (origin is Sequence or (origin is Union and any(get_origin(a) is Sequence for a in args))):
                setattr(self, field.name, [float(x) for x in value])



@dataclass(slots=True)
class PipelineOptions:
    seed: int
    prompt: str
    negative_prompt: Optional[str] = None 
    guidance_scale: float = 1.0
    guidance_scale2: float = 1.0
    steps: int = 50
    steps2: int = 0
    height: int = 0
    width: int = 0
    frames: int = 0
    frame_rate: float = 0.0
    strength: float = 1.0
    control_net_scale: float = 1.0
    lora_options: Optional[Sequence[LoraOption]] = None 
    scheduler_options: SchedulerOptions = None
    temp_filename: str = None
    frame_chunk: int = 0
    frame_chunk_overlap: int = 0
    noise_condition: int = 0
    enable_vae_tiling: bool = False
    enable_vae_slicing: bool = False

    def __post_init__(self):
        self.guidance_scale = float(self.guidance_scale)
        self.guidance_scale2 = float(self.guidance_scale2)
        self.frame_rate = float(self.frame_rate)
        self.strength = float(self.strength)
        self.control_net_scale = float(self.control_net_scale)
        self.frame_chunk = self.frames if self.frame_chunk == 0 else self.frame_chunk
        if (self.scheduler_options is not None and isinstance(self.scheduler_options, dict)):
            self.scheduler_options = SchedulerOptions(**self.scheduler_options)
        if (self.lora_options is not None and isinstance(self.lora_options, Sequence)):
            self.lora_options = [LoraOption(**dict(cfg)) for cfg in self.lora_options or []]
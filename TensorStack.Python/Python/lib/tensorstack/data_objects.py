from dataclasses import dataclass, field
from typing import Optional, Union, Sequence
import torch

def get_data_type(dtype: str):
    if dtype == "float8_e5m2":
        return torch.float8_e5m2
    if dtype == "float8_e4m3fn":
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
    if dtype == "float8":
        return torch.float8_e4m3fn
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

    data_type: Union[str, torch.dtype] = "bfloat16"
    quant_data_type: Union[str, torch.dtype] = "bfloat16"

    # HF / loading
    variant: Optional[str] = None
    cache_directory: Optional[str] = None
    secure_token: Optional[str] = None

    lora_adapters: Optional[Sequence[LoraConfig]] = None
    control_net: Optional[ControlNetConfig] = None
    checkpoint_config: Optional[CheckpointConfig] = None
  
    def __post_init__(self):
        self.data_type = get_data_type(self.data_type)
        self.quant_data_type = get_data_type(self.quant_data_type)
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


        model_ckpt = getattr(self.checkpoint_config, "model_checkpoint", None)
        self.is_gguf = (
            (model_ckpt is not None and str(model_ckpt).lower().endswith(".gguf")) or
            (getattr(self, "base_model_path", None) is not None and str(self.base_model_path).lower().endswith(".gguf"))
        )


@dataclass(slots=True)
class SchedulerOptions:
    # Core
    num_train_timesteps: int = 1000
    steps_offset: int = 0

    # IsTimestep
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"        # BetaScheduleType
    prediction_type: str = "epsilon"            # PredictionType
    variance_type: Optional[str] = None          # VarianceType
    timestep_spacing: str = "linspace"           # TimestepSpacingType

    # IsClipSample
    clip_sample: bool = False
    clip_sample_range: float = 1.0

    # IsThreshold
    thresholding: bool = False
    dynamic_thresholding_ratio: float = 0.995
    sample_max_value: float = 1.0

    # IsKarras
    use_karras_sigmas: bool = False
    sigma_min: Optional[float] = None
    sigma_max: Optional[float] = None
    rho: float = 7.0

    # IsMultiStep
    solver_order: int = 2
    solver_type: str = "midpoint"                # SolverType
    algorithm_type: str = "dpmsolver++"          # AlgorithmType
    lower_order_final: bool = True

    # IsStochastic
    eta: float = 0.0
    s_noise: float = 1.0
    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = 0.0   # 0 == +inf

    # IsFlowMatch
    shift: float = 1.0
    use_dynamic_shifting: bool = False
    base_shift: float = 0.5
    max_shift: float = 1.15
    stochastic_sampling: bool = False

    flow_shift: float = 1.0

    # Sequence lengths
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096

    def __post_init__(self):
        self.beta_start = float(self.beta_start)
        self.beta_end = float(self.beta_end)
        self.clip_sample_range = float(self.clip_sample_range)
        self.dynamic_thresholding_ratio = float(self.dynamic_thresholding_ratio)
        self.sample_max_value = float(self.sample_max_value)
        self.rho = float(self.rho)
        self.eta = float(self.eta)
        self.s_noise = float(self.s_noise)
        self.s_churn = float(self.s_churn)
        self.s_tmin = float(self.s_tmin)
        self.s_tmax = float(self.s_tmax)
        self.shift = float(self.shift)
        self.base_shift = float(self.base_shift)
        self.max_shift = float(self.max_shift)
        self.flow_shift = float(self.flow_shift)
        if self.s_tmax == 0.0:
            self.s_tmax = float("infinity")



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
    scheduler: str = "ddim"
    lora_options: Optional[Sequence[LoraOption]] = None 
    scheduler_options: SchedulerOptions = None
    temp_filename: str = None

    def __post_init__(self):
        self.guidance_scale = float(self.guidance_scale)
        self.guidance_scale2 = float(self.guidance_scale2)
        self.frame_rate = float(self.frame_rate)
        self.strength = float(self.strength)
        self.control_net_scale = float(self.control_net_scale)
        if (self.scheduler_options is not None and isinstance(self.scheduler_options, dict)):
            self.scheduler_options = SchedulerOptions(**self.scheduler_options)
        if (self.lora_options is not None and isinstance(self.lora_options, Sequence)):
            self.lora_options = [LoraOption(**dict(cfg)) for cfg in self.lora_options or []]
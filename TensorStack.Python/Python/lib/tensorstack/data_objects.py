from dataclasses import dataclass, field
from typing import Optional, Union
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
class PipelineConfig:
    # Required / core
    base_model_path: str
    pipeline: str
    process_type: str
    memory_mode: str

    # Optional
    control_net_path: Optional[str] = None

    # Device
    device: str = "cuda"
    device_id: int = 0

    data_type: Union[str, torch.dtype] = "bfloat16"
    quant_data_type: Union[str, torch.dtype] = "bfloat16"

    # HF / loading
    variant: Optional[str] = None
    cache_directory: Optional[str] = None
    secure_token: Optional[str] = None

    checkpoint_config: Optional[CheckpointConfig] = None
  
    def __post_init__(self):
        self.data_type = get_data_type(self.data_type)
        self.quant_data_type = get_data_type(self.quant_data_type)
        if (self.checkpoint_config is not None and isinstance(self.checkpoint_config, dict)):
            self.checkpoint_config = CheckpointConfig(**self.checkpoint_config)
        elif self.checkpoint_config is None:
            self.checkpoint_config = CheckpointConfig()


@dataclass(slots=True)
class LoraConfig:
    path: str
    name: str
    weights: str


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

    flow_shift: float = 1

    # Sequence lengths
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096

    def __post_init__(self):
        if self.s_tmax == 0.0:
            self.s_tmax = float("infinity")



import torch
from typing import Any
import tensorstack.data_objects as DataObjects
from transformers import (
    TorchAoConfig as TransformersTorchAoConfig,
    QuantoConfig as TransformersQuantoConfig
)
from diffusers import (
    PipelineQuantizationConfig,
    TorchAoConfig as DiffusersTorchAoConfig,
    QuantoConfig as DiffusersQuantoConfig,
    GGUFQuantizationConfig as DiffusersGGUFConfig
)

try:
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    _HAS_TORCHAO = True
except Exception:
    quantize_ = None
    Int8WeightOnlyConfig = None
    _HAS_TORCHAO = False


try:
    from optimum.quanto import freeze, qint8, qfloat8, quantize
    _HAS_QUANTO = True
except Exception:
    freeze = None
    qint8 = None
    qfloat8 = None
    quantize = None
    _HAS_QUANTO = False


print(f"[Quantize] Initialize quantization backend, torchao: {_HAS_TORCHAO}, quanto: {_HAS_QUANTO}")


def get_quantize_model_config(dtype: torch.dtype, quant_dtype: torch.dtype, memory_mode: str):
    if not is_quantization_supported(dtype, quant_dtype, memory_mode):
        return None, None

    if _HAS_TORCHAO:
        if quant_dtype == torch.int8:
            print(f"[Quantize] Quantizing model from '{dtype}' to '{quant_dtype}'")
            return DiffusersTorchAoConfig(Int8WeightOnlyConfig()), TransformersTorchAoConfig(Int8WeightOnlyConfig())
        
    elif _HAS_QUANTO:
        if quant_dtype == torch.int8:
            print(f"[Quantize] Quantizing model from '{dtype}' to '{quant_dtype}'")
            return DiffusersQuantoConfig(weights_dtype="int8"), TransformersQuantoConfig(weights_dtype="int8")
        elif quant_dtype == torch.float8_e4m3fn:
            print(f"[Quantize] Quantizing model from '{dtype}' to '{quant_dtype}'")
            return DiffusersQuantoConfig(weights_dtype="float8"), TransformersQuantoConfig(weights_dtype="float8")
                
    return None, None


def get_single_file_config(config: DataObjects.PipelineConfig):
    return DiffusersGGUFConfig(compute_dtype=config.data_type) if config.is_gguf else None


def quantize_model(config: DataObjects.PipelineConfig, model: Any):
    if config.is_gguf:
        return

    if not is_quantization_supported(model.dtype, config.quant_data_type, config.memory_mode):
        return

    if _HAS_TORCHAO:
        if config.quant_data_type == torch.int8:
            print(f"[Quantize] Quantizing model from '{model.dtype}' to '{config.quant_data_type}'")
            quantize_(model, Int8WeightOnlyConfig())

    elif _HAS_QUANTO:
        if config.quant_data_type == torch.int8:
            print(f"[Quantize] Quantizing model from '{model.dtype}' to '{config.quant_data_type}'")
            quantize(model, weights=qint8)
            freeze(model)
        elif config.quant_data_type == torch.float8_e4m3fn:
            print(f"[Quantize] Quantizing model from '{model.dtype}' to '{config.quant_data_type}'")
            quantize(model, weights=qfloat8)
            freeze(model)


def is_quantization_supported(dtype: torch.dtype, quant_dtype: torch.dtype, memory_mode: str) -> bool:
    if not (_HAS_TORCHAO or _HAS_QUANTO):
        print(f"[Quantize] No quantization backend found.")
        return False
    if memory_mode == "OffloadCPU":
        print(f"[Quantize] OffloadCPU does not support quantization.")
        return False
    if quant_dtype == dtype:
        print(f"[Quantize] Model is already '{quant_dtype}' skipping quantization.")
        return False
    return True
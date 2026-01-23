import torch
from typing import Any
from transformers import (
    TorchAoConfig as TransformersTorchAoConfig,
    QuantoConfig as TransformersQuantoConfig
)
from diffusers import (
    PipelineQuantizationConfig,
    TorchAoConfig as DiffusersTorchAoConfig,
    QuantoConfig as DiffusersQuantoConfig
)

try:
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    _HAS_TORCHAO = True
except Exception:
    quantize_ = None
    Int8WeightOnlyConfig = None
    _HAS_TORCHAO = False


try:
    from optimum.quanto import freeze, qint8, quantize
    _HAS_QUANTO = True
except Exception:
    freeze = None
    qint8 = None
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
                
    return None, None


def quantize_model(model: Any, quant_dtype: torch.dtype, memory_mode: str):
    if not is_quantization_supported(model.dtype, quant_dtype, memory_mode):
        return

    if _HAS_TORCHAO:
        if quant_dtype == torch.int8:
            print(f"[Quantize] Quantizing model from '{model.dtype}' to '{quant_dtype}'")
            quantize_(model, Int8WeightOnlyConfig())
    elif _HAS_QUANTO:
        if quant_dtype == torch.int8:
            print(f"[Quantize] Quantizing model from '{model.dtype}' to '{quant_dtype}'")
            quantize(model, weights=qint8)
            freeze(model)


def get_quantize_pipeline_config(
        dtype: torch.dtype, 
        quant_dtype: torch.dtype,
        memory_mode: str,
        diffusers: list[str],
        transformers: list[str]
    ):
    if not is_quantization_supported(dtype, quant_dtype, memory_mode):
        return None

    quant_mapping = {}
    if quant_dtype == torch.int8:
        diffusers_cfg, transformers_cfg = get_quantize_model_config(dtype, quant_dtype, memory_mode)
        if diffusers_cfg is None and transformers_cfg is None:
            return None

        for name in diffusers:
            quant_mapping[name] = diffusers_cfg

        for name in transformers:
            quant_mapping[name] = transformers_cfg

    if not quant_mapping:
        return None
    
    print(f"[Quantize] Quantizing model from '{dtype}' to '{quant_dtype}'")
    return PipelineQuantizationConfig(quant_mapping=quant_mapping)


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
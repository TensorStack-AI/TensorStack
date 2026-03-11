import torch
from typing import Any
import tensorstack.data_objects as DataObjects
from transformers import (
    QuantoConfig as TransformersQuantoConfig,
    BitsAndBytesConfig as TransformersBitsAndBytesConfig
)
from diffusers import (
    QuantoConfig as DiffusersQuantoConfig,
    GGUFQuantizationConfig as DiffusersGGUFConfig,
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig
)


try:
    from optimum.quanto import freeze, qint8, qfloat8, quantize, qint4
    _HAS_QUANTO = True
except Exception:
    freeze = None
    qint4 = None
    qint8 = None
    qfloat8 = None
    quantize = None
    _HAS_QUANTO = False
    

try:
    import bitsandbytes
    _HAS_BITSANDBYTES = True
except Exception:
    _HAS_BITSANDBYTES = False


print(f"[Quantize] Initialize quantization backend, quanto: {_HAS_QUANTO}, bitsandbytes: {_HAS_BITSANDBYTES}")


def get_quantize_model_config(dtype: torch.dtype, quant_dtype: torch.dtype, memory_mode: str):
    if not is_quantization_supported(dtype, quant_dtype, memory_mode):
        return None, None
        
    elif _HAS_QUANTO:
        if quant_dtype == torch.int:
            print(f"[Quantize] Quantizing model from '{dtype}' to 'int4'")
            return DiffusersQuantoConfig(weights_dtype="int4"), TransformersQuantoConfig(weights_dtype="int4")
        elif quant_dtype == torch.int8:
            print(f"[Quantize] Quantizing model from '{dtype}' to 'int8'")
            return DiffusersQuantoConfig(weights_dtype="int8"), TransformersQuantoConfig(weights_dtype="int8")
        elif quant_dtype == torch.float8_e4m3fn:
            print(f"[Quantize] Quantizing model from '{dtype}' to 'float8'")
            return DiffusersQuantoConfig(weights_dtype="float8"), TransformersQuantoConfig(weights_dtype="float8")
        
    elif _HAS_BITSANDBYTES:
        if quant_dtype == torch.int:
            print(f"[Quantize] Quantizing model from '{dtype}' to 'int4'")
            return DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype), TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        elif quant_dtype == torch.int8:
            print(f"[Quantize] Quantizing model from '{dtype}' to 'int8'")
            return DiffusersBitsAndBytesConfig(load_in_8bit=True), TransformersBitsAndBytesConfig(load_in_8bit=True)
                
    return None, None



def get_single_file_config(config: DataObjects.PipelineConfig):
    if config.is_gguf:
        return DiffusersGGUFConfig(compute_dtype=config.data_type)
    
    elif _HAS_BITSANDBYTES:
        if config.quant_dtype == torch.int8:
            return DiffusersBitsAndBytesConfig(load_in_8bit=True)
        elif config.quant_dtype == torch.float8_e4m3fn:
            return DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=config.data_type)



def quantize_model(config: DataObjects.PipelineConfig, model: Any):
    if config.is_gguf or _HAS_BITSANDBYTES:
        return

    if not is_quantization_supported(model.dtype, config.quant_data_type, config.memory_mode):
        return

    elif _HAS_QUANTO:
        if config.quant_data_type == torch.int:
            print(f"[Quantize] Quantizing model from '{model.dtype}' to 'int4'")
            quantize(model, weights=qint4)
            freeze(model)
        elif config.quant_data_type == torch.int8:
            print(f"[Quantize] Quantizing model from '{model.dtype}' to 'int8'")
            quantize(model, weights=qint8)
            freeze(model)
        elif config.quant_data_type == torch.float8_e4m3fn:
            print(f"[Quantize] Quantizing model from '{model.dtype}' to 'float8'")
            quantize(model, weights=qfloat8)
            freeze(model)


def is_quantization_supported(dtype: torch.dtype, quant_dtype: torch.dtype, memory_mode: str) -> bool:
    if not (_HAS_QUANTO or _HAS_BITSANDBYTES):
        print(f"[Quantize] No quantization backend found.")
        return False
    if memory_mode == "OffloadCPU":
        print(f"[Quantize] OffloadCPU does not support quantization.")
        return False
    if quant_dtype == dtype:
        print(f"[Quantize] Model is already '{quant_dtype}' skipping quantization.")
        return False
    return True
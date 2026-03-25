import torch
from typing import Any
import tensorstack.data_objects as DataObjects
from tensorstack.enums import QuantType, QuantBackend, QuantTarget, MemoryMode
import bitsandbytes
from optimum.quanto import freeze, qfloat8, quantize, qint4
from transformers import (
    QuantoConfig as TransformersQuantoConfig,
    BitsAndBytesConfig as TransformersBitsAndBytesConfig
)
from diffusers import (
    QuantoConfig as DiffusersQuantoConfig,
    GGUFQuantizationConfig as DiffusersGGUFConfig,
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
)


#------------------------------------------------
# Quantize a PyTorch model
#------------------------------------------------
def quantize_model(config: DataObjects.PipelineConfig, model: Any, is_gguf: bool):
    if config.memory_mode == MemoryMode.OffloadCPU:
        print(f"[Quantize] OffloadCPU not supported")
        return None
    if is_gguf:
        return

    data_type = config.data_type
    quant_type = config.quant_type
    if quant_type == QuantType.Q16Bit:
        print(f"[Quantize] {quant_type} not supported")
        return
    elif quant_type == QuantType.Q8Bit:
        print(f"[Quantize] {QuantBackend.QUANTO}, {data_type} -> {quant_type}")
        quantize(model, weights=qfloat8)
        freeze(model)
    elif quant_type == QuantType.Q4Bit:
        print(f"[Quantize] {QuantBackend.QUANTO}, {data_type} -> {quant_type}")
        quantize(model, weights=qint4)
        freeze(model)



#------------------------------------------------
# Auto Quantization Configuration for from_pretrained
#------------------------------------------------
def auto_pretrained_config(config: DataObjects.PipelineConfig, target: QuantTarget):
    if config.memory_mode ==  MemoryMode.OffloadCPU:
        print(f"[Quantize] OffloadCPU not supported")
        return None

    data_type = config.data_type
    quant_type = config.quant_type
    if quant_type == QuantType.Q16Bit:
        return pretrained_config(target, QuantBackend.NONE, QuantType.Q16Bit, data_type)
    elif target == QuantTarget.TEXT_ENCODER:
        if quant_type == QuantType.Q8Bit:
            return pretrained_config(target, QuantBackend.QUANTO, QuantType.Q8Bit, data_type)
        elif quant_type == QuantType.Q4Bit:
            return pretrained_config(target, QuantBackend.BITSANDBYTES, QuantType.Q4Bit, data_type)

    elif target == QuantTarget.TRANSFORMER:
        if quant_type == QuantType.Q8Bit:
            return pretrained_config(target, QuantBackend.QUANTO, QuantType.Q8Bit, data_type)
        elif quant_type == QuantType.Q4Bit:
            return pretrained_config(target, QuantBackend.BITSANDBYTES, QuantType.Q4Bit, data_type)

    return None



#------------------------------------------------
# Quantization Configuration for from_pretrained
#------------------------------------------------
def pretrained_config(target: QuantTarget, backend: QuantBackend, quant_type: QuantType, compute_type: torch.dtype):
    if quant_type == QuantType.Q16Bit or backend == QuantBackend.NONE:
        print(f"[Quantize] {quant_type} not supported")
        return None

    print(f"[Quantize] {target}, {backend}, {compute_type} -> {quant_type}")
    if backend == QuantBackend.QUANTO:
        if target == QuantTarget.TEXT_ENCODER:
            if quant_type == QuantType.Q8Bit:
                return TransformersQuantoConfig(weights_dtype="float8") # "int8"
            elif quant_type == QuantType.Q4Bit:
                return TransformersQuantoConfig(weights_dtype="int4")

        elif target == QuantTarget.TRANSFORMER:
            if quant_type == QuantType.Q8Bit:
                return DiffusersQuantoConfig(weights_dtype="float8")  # "int8"
            elif quant_type == QuantType.Q4Bit:
                return DiffusersQuantoConfig(weights_dtype="int4")

    elif backend == QuantBackend.BITSANDBYTES:
        if target == QuantTarget.TEXT_ENCODER:
            if quant_type == QuantType.Q8Bit:
                return TransformersBitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
            elif quant_type == QuantType.Q4Bit:
                return TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=compute_type, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

        elif target == QuantTarget.TRANSFORMER:
            if quant_type == QuantType.Q8Bit:
                return DiffusersBitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
            elif quant_type == QuantType.Q4Bit:
                return DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=compute_type, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

    return None



#------------------------------------------------
# Auto Quantization Configuration for from_single_file
#------------------------------------------------
def auto_single_file_config(config: DataObjects.PipelineConfig, target: QuantTarget, is_gguf: bool):
    if config.memory_mode ==  MemoryMode.OffloadCPU:
        print(f"[Quantize] OffloadCPU not supported")
        return None

    if is_gguf:
        return DiffusersGGUFConfig(compute_dtype=config.data_type)
    return None



#------------------------------------------------
# Quantization Configuration for from_single_file
#------------------------------------------------
def single_file_config(target: QuantTarget, backend: QuantBackend, quant_type: QuantType, compute_type: torch.dtype, is_gguf: bool):
    if is_gguf:
        return DiffusersGGUFConfig(compute_dtype=compute_type)

    return pretrained_config(target, backend, quant_type, compute_type)
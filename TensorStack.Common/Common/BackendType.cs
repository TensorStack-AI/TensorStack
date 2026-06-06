using System.ComponentModel.DataAnnotations;

namespace TensorStack.Common
{
    public enum BackendType
    {
        [Display(Name = "OnnxRuntime", ShortName = "Onnx", Description = "OnnxRuntime .NET model inference using TensorStack")]
        OnnxRuntime = 0,

        [Display(Name = "PyTorch", ShortName = "Onnx", Description = "PyTorch model inference using HuggingFace Diffusers & Transformers")]
        PyTorch = 10
    }
}

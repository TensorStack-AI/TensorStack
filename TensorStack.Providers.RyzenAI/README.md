# TensorStack.Providers.RyzenAI

## Important External Binaries Required !!

The RyzenAI binaries are needed for the `Provider` to work correctly, you can download the required files from https://huggingface.co/amd/stable-diffusion-1.5-amdnpu/tree/main/libs
- ryzen_mm.dll
- ryzenai_onnx_utils.dll
- onnx_custom_ops.dll

Place these files in `TensorStack.Providers.RyzenAI` folder

---

### Basic Initialization

```csharp

var provider = Provider.GetProvider(DeviceType.NPU);

```

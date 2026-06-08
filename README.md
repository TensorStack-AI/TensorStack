# TensorStack

A modular .NET SDK for building AI-powered applications.

TensorStack provides reusable components for local AI inference, text generation, image generation, media processing, hardware acceleration, and application development through a unified developer experience.

Built for performance and flexibility, TensorStack powers all applications developed by TensorStack-AI, including [**Amuse**](https://github.com/TensorStack-AI/AmuseAI), the official reference application and UI demonstration of the platform.

---

## Features

- Native .NET SDK
- Local AI inference
- Text generation and LLM integration
- Stable Diffusion image generation
- Image upscaling and extraction pipelines
- Image, video, and audio processing
- Hardware acceleration support
- Python interoperability
- Extensible provider architecture
- Cross-platform development
---

## Packages

### AI & Inference

| Package | Description | Docs | Package |
|---------|-------------|------|---------|
| TensorStack.Common | Shared runtime and utilities | [README](TensorStack.Common/README.md) | [![Common Badge](https://img.shields.io/nuget/v/TensorStack.Common?color=4bc51e&label=TensorStack.Common)](https://www.nuget.org/packages/TensorStack.Common) | 
| TensorStack.TextGeneration | Text generation and LLM integrations | [README](TensorStack.TextGeneration/README.md) | [![TextGeneration Badge](https://img.shields.io/nuget/v/TensorStack.TextGeneration?color=4bc51e&label=TensorStack.TextGeneration)](https://www.nuget.org/packages/TensorStack.TextGeneration) |
| TensorStack.StableDiffusion | Stable Diffusion image generation | [README](TensorStack.StableDiffusion/README.md) | [![StableDiffusion Badge](https://img.shields.io/nuget/v/TensorStack.StableDiffusion?color=4bc51e&label=TensorStack.StableDiffusion)](https://www.nuget.org/packages/TensorStack.StableDiffusion) |
| TensorStack.Upscaler | AI image upscaling | [README](TensorStack.Common/Upscaler.md) | [![Upscaler Badge](https://img.shields.io/nuget/v/TensorStack.Upscaler?color=4bc51e&label=TensorStack.Upscaler)](https://www.nuget.org/packages/TensorStack.Upscaler) |
| TensorStack.Extractors | Feature extraction and analysis |[README](TensorStack.Common/Extractors.md) | [![Upscaler Badge](https://img.shields.io/nuget/v/TensorStack.Extractors?color=4bc51e&label=TensorStack.Extractors)](https://www.nuget.org/packages/TensorStack.Extractors) |
| TensorStack.Python | Python interoperability | [README](TensorStack.Common/Python.md) | [![Python Badge](https://img.shields.io/nuget/v/TensorStack.Python?color=4bc51e&label=TensorStack.Python)](https://www.nuget.org/packages/TensorStack.Python) |

### Inference Providers

| Package | Description | Docs | Package |
|---------|-------------|------|---------|
| TensorStack.Providers.CPU | CPU execution provider | [README](TensorStack.Providers.CPU/README.md) | [![CPU Badge](https://img.shields.io/nuget/v/TensorStack.Providers.CPU?color=4bc51e&label=TensorStack.Providers.CPU)](https://www.nuget.org/packages/TensorStack.Providers.CPU) |
| TensorStack.Providers.CUDA | NVIDIA CUDA execution provider | [README](TensorStack.Providers.CUDA/README.md) | [![CUDA Badge](https://img.shields.io/nuget/v/TensorStack.Providers.CUDA?color=4bc51e&label=TensorStack.Providers.CUDA)](https://www.nuget.org/packages/TensorStack.Providers.CUDA) |
| TensorStack.Providers.DML | DirectML execution provider | [README](TensorStack.Providers.DML/README.md) | [![DML Badge](https://img.shields.io/nuget/v/TensorStack.Providers.DML?color=4bc51e&label=TensorStack.Providers.DML)](https://www.nuget.org/packages/TensorStack.Providers.DML) |

---

### Media Processing

#### Image

| Package | Description | Docs | Package |
|---------|-------------|------|---------|
| TensorStack.Image | Shared image utilities | [README](TensorStack.Image/README.md) | [![Image Badge](https://img.shields.io/nuget/v/TensorStack.Image?color=4bc51e&label=TensorStack.Image)](https://www.nuget.org/packages/TensorStack.Image) |
| TensorStack.Image.Bitmap | Windows Forms `Bitmap` ↔ Tensor conversion | [README](TensorStack.Image.Bitmap/README.md) | [![Bitmap Badge](https://img.shields.io/nuget/v/TensorStack.Image.Bitmap?color=4bc51e&label=TensorStack.Image.Bitmap)](https://www.nuget.org/packages/TensorStack.Image.Bitmap) |
| TensorStack.Image.BitmapImage |  WPF `BitmapImage` ↔ Tensor conversion | [README](TensorStack.Image.BitmapImage/README.md) | [![BitmapImage Badge](https://img.shields.io/nuget/v/TensorStack.Image.BitmapImage?color=4bc51e&label=TensorStack.Image.BitmapImage)](https://www.nuget.org/packages/TensorStack.Image.BitmapImage) |
| TensorStack.Image.ImageSharp | ImageSharp `Image` ↔ Tensor conversion | [README](TensorStack.Image.ImageSharp/README.md) | [![ImageSharp Badge](https://img.shields.io/nuget/v/TensorStack.Image.ImageSharp?color=4bc51e&label=TensorStack.Image.ImageSharp)](https://www.nuget.org/packages/TensorStack.Image.ImageSharp) |

#### Video

| Package | Description | Docs | Package |
|---------|-------------|------|---------|
| TensorStack.Video | Shared video utilities | [README](TensorStack.Video/README.md) | [![Video Badge](https://img.shields.io/nuget/v/TensorStack.Video?color=4bc51e&label=TensorStack.Video)](https://www.nuget.org/packages/TensorStack.Video) |
| TensorStack.Video.Windows | Windows implementation using OpenCvSharp4 | [README](TensorStack.Video.Windows/README.md) | [![Video.Windows Badge](https://img.shields.io/nuget/v/TensorStack.Video.Windows?color=4bc51e&label=TensorStack.Video.Windows)](https://www.nuget.org/packages/TensorStack.Video.Windows) |

#### Audio

| Package | Description | Docs | Package |
|---------|-------------|------|---------|
| TensorStack.Audio | Shared audio utilities | [README](TensorStack.Audio/README.md) | [![Audio Badge](https://img.shields.io/nuget/v/TensorStack.Audio?color=4bc51e&label=TensorStack.Audio)](https://www.nuget.org/packages/TensorStack.Audio) |
| TensorStack.Audio.Windows | Windows implementation using FFMPEG | [README](TensorStack.Audio.Windows/README.md) | [![Audio.Windows Badge](https://img.shields.io/nuget/v/TensorStack.Audio.Windows?color=4bc51e&label=TensorStack.Audio.Windows)](https://www.nuget.org/packages/TensorStack.Audio.Windows) |

---

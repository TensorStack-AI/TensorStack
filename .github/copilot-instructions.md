# TensorStack Copilot Instructions

TensorStack is a modular .NET framework for high-performance ONNX-based AI model inference across multiple hardware providers and media types (image, video, audio).

## Architecture Overview

**Layered Design**: The codebase follows a strict layering pattern:
- **TensorStack.Common**: Core abstractions - `IPipeline<T,O>`, tensor types, model session management, device/execution provider registry
- **Media Layer** (`TensorStack.Image*`, `TensorStack.Audio*`, `TensorStack.Video*`): Platform-specific input handling (WinForms Bitmap, BitmapImage, ImageSharp, Windows audio/video APIs)
- **Provider Layer** (`TensorStack.Providers.*`): Execution providers wrap ONNX Runtime's `SessionOptions` - CPU, CUDA, DML each expose a `Provider.GetProvider()` static method
- **Feature Pipelines** (`TensorStack.Extractors`, `TensorStack.Upscaler`, `TensorStack.TextGeneration`, `TensorStack.StableDiffusion`): Concrete `IPipeline` implementations

**Key Integration Points**:
- Pipelines consume `ModelConfig` (path + ExecutionProvider) and produce typed outputs via `RunAsync()`
- `DeviceManager` is a static registry initialized with an ExecutionProvider and optional native library path
- `ModelSession<T>` manages ONNX Runtime's `InferenceSession` lifecycle and `SessionOptions` configuration

## Essential Patterns

### Pipeline Implementation
Every feature pipeline implements `IPipeline<OutputType, OptionsType>`:
```csharp
public class MyPipeline : IPipeline<OutputTensor, MyOptions>
{
    public async Task LoadAsync(CancellationToken ct) => await _model.LoadAsync(ct);
    public async Task<OutputTensor> RunAsync(MyOptions opts, IProgress<RunProgress> progress = null, CancellationToken ct = default)
    {
        var result = await _model.InferAsync(opts.Input);
        progress?.Report(new RunProgress(RunProgress.GetTimestamp()));
        return result;
    }
}
```
- Always implement `IDisposable` to release ONNX sessions
- Use `IPipelineStream<T, O>` for streaming video frame processing with `IAsyncEnumerable<T>`
- Progress callbacks report elapsed time via `RunProgress.GetTimestamp()`

### Model Configuration Pattern
Inherit from `ModelConfig` to define model-specific settings:
```csharp
public record ExtractorConfig : ModelConfig
{
    public Normalization Normalization { get; set; }
    public Normalization OutputNormalization { get; set; }
    public bool IsDynamicOutput { get; set; }
}
```
- Always use `record` for immutability
- Set execution provider via `config.SetProvider(executionProvider)` or in initialization
- Path must point to a valid ONNX file

### Tensor Type Conversions
`ImageTensor` and `VideoTensor` inherit from `Tensor<float>` with automatic shape handling:
- `ImageTensor`: Always dims `[1, channels, height, width]`; auto-validates 1, 3, or 4 channel images
- `VideoTensor`: Dims `[batch, frames, channels, height, width]`
- Use `tensor.AsSpan()` to access underlying data; dimensions available via `tensor.Dimensions`

## Build & Release Workflow

**Individual Package Build**:
```powershell
dotnet build TensorStack.Extractors/TensorStack.Extractors.csproj -c Release
dotnet pack TensorStack.Extractors/TensorStack.Extractors.csproj -c Release
```

**Batch Release** (automated via `BuildRelease.bat`):
- Builds all projects in dependency order: Common → Media (Bitmap, Audio, Video) → Providers → Features
- Packs each as a NuGet package to `./Nuget/` directory
- Version centralized in `Directory.Build.props` (currently 0.4.0)

## Platform-Specific Conventions

**Windows Media APIs**: `TensorStack.Image.Bitmap`, `TensorStack.Audio.Windows`, `TensorStack.Video.Windows` wrap WinForms/Windows Media Foundation:
- Abstract base classes (`ImageInputBase`, `AudioInputBase`) in platform-agnostic projects
- Concrete implementations (e.g., `ImageInput`) in Windows-specific projects
- Use `.targets` files (see `TensorStack.Audio.Windows.targets`) to define native dependencies

**Provider Selection**: Each provider has a static `Provider` class:
```csharp
ExecutionProvider provider = TensorStack.Providers.CPU.Provider.GetProvider();
DeviceManager.Initialize(provider, "TensorStack");
```

## Critical Developer Tasks

**Adding a New Pipeline**:
1. Create `TensorStack.MyFeature/` folder with `.csproj`
2. Define `MyConfig : ModelConfig` and `MyOptions : IRunOptions`
3. Implement concrete pipeline class inheriting `IPipeline<OutputType, MyOptions>`
4. Add model loading via `ModelSession<MyConfig>` in the pipeline
5. Reference `TensorStack.Common` and required media packages
6. Add build+pack steps to `BuildRelease.bat`

**Supporting a New Execution Provider**:
1. Create `TensorStack.Providers.NewProvider/` with `Provider.cs`
2. Implement static `GetProvider()` returning `ExecutionProvider` with proper `SessionOptions`
3. Register native library path in `DeviceManager.Initialize()`
4. Each provider's `SessionOptions` factory is stored in the `ExecutionProvider` constructor

**Cross-Cutting Concerns**: Use extensions in each module's `Common/` folder (e.g., `TensorStack.Extractors/Common/`) for normalization, model-specific tensor conversions, and pipeline-specific utilities.

## NuGet & Dependencies

- **Core Dependency**: `Microsoft.ML.OnnxRuntime` (abstracts ONNX inference)
- **Media**: Individual packages for Bitmap, BitmapImage, ImageSharp, Windows audio/video
- **Platform**: .NET 8+ target framework; Windows-specific projects use Windows SDK APIs
- **No External Model Loading**: Models are referenced by file path; users supply ONNX files

## Nullable & Implicit Usings Policy

- `Directory.Build.props` disables `ImplicitUsings` and `Nullable` globally
- Always include full namespace imports explicitly
- Null checks use `ArgumentNullException.ThrowIfNull()` pattern

---

**Reference Examples**: See [TensorStack.Extractors/Pipelines/ExtractorPipeline.cs](../../TensorStack.Extractors/Pipelines/ExtractorPipeline.cs), [TensorStack.Common/Pipeline/IPipeline.cs](../../TensorStack.Common/Pipeline/IPipeline.cs), [TensorStack.Providers.CPU/Provider.cs](../../TensorStack.Providers.CPU/Provider.cs)

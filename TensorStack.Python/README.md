# TensorStack.Python

C# => Python inferface

## External Packages
`CSnakes.Runtime` - `v2.0.0-dev.251218-1221` (TensorStack.Python\Packages)

## Virtual Environment 
The `PythonService` can be used to download and install Python and create virtual environments
```csharp
// Virtual Environment Config
var serverConfig = new EnvironmentConfig
{
    Environment = "default-cuda",
    Directory = "PythonRuntime",
    Requirements =
    [
        "--extra-index-url https://download.pytorch.org/whl/cu118",
        "torch==2.7.0+cu118",
        "typing",
        "wheel",
        "transformers",
        "accelerate",
        "diffusers",
        "protobuf",
        "sentencepiece",
        "pillow",
        "ftfy",
        "scipy",
        "peft"
    ]
};

// PythonManager
var pythonService = new PythonManager(serverConfig, PipelineProgress.ConsoleCallback);

// Create/Load Virtual Environment
await pythonService.CreateEnvironmentAsync();
```

## Python Pipelines
Once you have created a virtual environment you can now load a pipeline
```csharp
// Pipeline Config
var pipelineConfig = new PipelineConfig
{
    Path = "Tongyi-MAI/Z-Image-Turbo",
    Pipeline = "ZImagePipeline",
    IsModelOffloadEnabled = true,
    DataType = DataType.Bfloat16
};

// Create Pipeline Proxy
using (var pythonPipeline = new PipelineProxy(pipelineConfig))
{
    // Download/Load Model
    await pythonPipeline.LoadAsync();

    // Generate Option
    var options = new PipelineOptions
    {
        Prompt = "cute cat",
        Steps = 10,
        Width = 1024,
        Height = 1024,
        GuidanceScale = 0,
        Scheduler = SchedulerType.FlowMatchEulerDiscrete,
    };

    // Generate
    var response = await pythonPipeline
        .GenerateAsync(options)
        .WithPythonLogging(pythonPipeline, PipelineProgress.ConsoleCallback);

    // Save Image
    await response
        .AsImageTensor()
        .SaveAsync("Result.png");
}
```
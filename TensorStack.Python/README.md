# TensorStack.Python
C# => Python Diffusers Inferface

## Supported Diffusers Pipelines
### Chroma
- ChromaPipeline
- ChromaImg2ImgPipeline
### Qwen Image
- QwenImagePipeline
- QwenImageImg2ImgPipeline
### Wan Video
- WanPipeline
- WanImageToVideoPipeline
### Z-Image
- ZImagePipeline
- ZImageImg2ImgPipeline

## Python Virtual Environment 
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
    Path = "Qwen/Qwen-Image-Edit",
    Pipeline = "QwenImagePipeline",
    ProcessType = ProcessType.ImageEdit,
    IsFullOffloadEnabled = true,
    DataType = DataType.Bfloat16
};

// Create Pipeline Proxy
using (var pythonPipeline = new PythonPipeline(pipelineConfig, PipelineProgress.ConsoleCallback))
{
    // Download/Load Model
    await pythonPipeline.LoadAsync();

    // Generate Option
    var options = new PipelineOptions
    {
        Prompt = "Yarn art style",
        Steps = 30,
        Width = 1024,
        Height = 1024,
        GuidanceScale = 4f,
        Scheduler = SchedulerType.FlowMatchEulerDiscrete,
        ImageInput = new ImageInput("Image.png")
    };

    // Generate
    var response = await pythonPipeline.GenerateAsync(options);

    // Save Image
    await response
        .AsImageTensor()
        .SaveAsync("Result.png");
}
```
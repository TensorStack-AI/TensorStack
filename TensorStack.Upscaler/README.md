# TensorStack.Upscaler

## Upscale Models
Below is a small list of known/tested upscale models
* https://huggingface.co/wuminghao/swinir
* https://huggingface.co/rocca/swin-ir-onnx
* https://huggingface.co/Xenova/swin2SR-classical-sr-x2-64
* https://huggingface.co/Xenova/swin2SR-classical-sr-x4-64
* https://huggingface.co/Neus/GFPGANv1.4


# Image Example
```csharp
// Upscaler config
var config = new UpscalerConfig("swin-2sr-classical-64-x2.onnx", Provider.DirectML)
{
    ScaleFactor = 2,
    Normalization = Normalization.ZeroToOne
};

// Create Pipeline
using (var pipeline = UpscalePipeline.Create(config))
{
    // Load Pipeline
    await pipeline.LoadAsync();

    // Read input image
    var input = new ImageInput("Input.png");

    // Options
    var options = new UpscaleImageOptions(input);

    // Run Upscale Pipeline
    var outputTensor = await pipeline.RunAsync(options);

    // Save Output image
    await outputTensor.SaveAsync("Output.png");
}
```



# Video Example
```csharp
// Upscaler config
var config = new UpscalerConfig("swin-2sr-classical-64-x2.onnx", Provider.DirectML)
{
    ScaleFactor = 2,
    Normalization = Normalization.ZeroToOne
};

// Create Pipeline
using (var pipeline = UpscalePipeline.Create(config))
{
    // Load Pipeline
    await pipeline.LoadAsync();

    // Read input video
    var video = new VideoInput("Input.mp4");

    // Get video input
    var input = await video.GetTensorAsync();

    // Options
    var options = new UpscaleVideoOptions(input);

    // Run Upscale Pipeline
    var outputTensor = await pipeline.RunAsync(options);

    // Save Output video
    await outputTensor.SaveAync("Output.mp4");
}
```



# Video Stream Example
```csharp
// Upscaler config
var config = new UpscalerConfig("swin-2sr-classical-64-x2.onnx", Provider.DirectML)
{
    ScaleFactor = 2,
    Normalization = Normalization.ZeroToOne
};

// Create Pipeline
using (var pipeline = UpscalePipeline.Create(config))
{
    // Load Pipeline
    await pipeline.LoadAsync();

    // Read input video
    var video = new VideoInput("Input.mp4");

    // Get video stream
    var videoStream = video.GetStreamAsync();

    // Options
    var options = new UpscaleStreamOptions(videoStream);

    // Get Upscale stream
    videoStream = pipeline.RunAsync(options);

    // Save Video Steam
    await videoStream.SaveAync("Output.mp4");
}
```
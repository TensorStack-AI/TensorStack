# TensorStack.Extractors

### Canny
* https://huggingface.co/axodoxian/controlnet_onnx/resolve/main/annotators/canny.onnx

### Hed
* https://huggingface.co/axodoxian/controlnet_onnx/resolve/main/annotators/hed.onnx

### Depth
* https://huggingface.co/axodoxian/controlnet_onnx/resolve/main/annotators/depth.onnx
* https://huggingface.co/Xenova/depth-anything-large-hf/onnx/model.onnx
* https://huggingface.co/julienkay/sentis-MiDaS

### Background Removal
* https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx


# Image Example
```csharp
// Extractor config
var config = new ExtractorConfig("hed.onnx", Provider.DirectML);

// Create Pipeline
using (var pipeline = ExtractorPipeline.Create(config))
{
    // Load Pipeline
    await pipeline.LoadAsync();

    // Read input image
    var input = new ImageInput("Input.png");

    // Options
    var options = new ExtractorImageOptions(input);

    // Run Extractor Pipeline
    var outputTensor = await pipeline.RunAsync(options);

    // Save Output image
    await outputTensor.SaveAsync("Output.png");
}
```



# Video Example
```csharp
// Extractor config
var config = new ExtractorConfig("hed.onnx", Provider.DirectML);

// Create Pipeline
using (var pipeline = ExtractorPipeline.Create(config))
{
    // Load Pipeline
    await pipeline.LoadAsync();

    // Read input video
    var video = new VideoInput("Input.mp4");

    // Get video input
    var input = await video.GetTensorAsync();

    // Options
    var options = new ExtractorVideoOptions(input);

    // Run Extractor Pipeline
    var outputTensor = await pipeline.RunAsync(inputTensor);

    // Save Output video
    await outputTensor.SaveAync("Output.mp4");
}
```



# Video Stream Example
```csharp
// Extractor config
var config = new ExtractorConfig("hed.onnx", Provider.DirectML);

// Create Pipeline
using (var pipeline = ExtractorPipeline.Create(config))
{
    // Load Pipeline
    await pipeline.LoadAsync();

    // Read input video
    var video = new VideoInput("Input.mp4");

    // Get video stream
    var videoStream = video.GetStreamAsync();

    // Options
    var options = new ExtractorStreamOptions(videoStream);

    // Get Extractor stream
    videoStream = pipeline.RunAsync(options);

    // Save Video Steam
    await videoStream.SaveAync("Output.mp4");
}
```
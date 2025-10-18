# TensorStack.Extractors
High-performance ONNX-based feature extraction for AI workflows. Includes models for edge detection, depth estimation, background removal, and other visual analysis tasks — designed for seamless integration with image and video processing pipelines.


## Quick Start
This minimal example demonstrates how to extract depth from image and video using `TensorStack.Extractors`.

```csharp
[nuget: TensorStack.Extractors]
[nuget: TensorStack.Providers.DML]
[nuget: TensorStack.Image.Bitmap]
[nuget: TensorStack.Video.Windows]
[nuget: TensorStack.Audio.Windows]
[model: https://huggingface.co/TensorStack/TensorStack/resolve/main/Extractor/Depth.onnx]

static async Task QuickStartAsync()
{
    // 1. Create the Extractor Pipeline
    var pipeline = ExtractorPipeline.Create(new ExtractorConfig
    {
        IsDynamicOutput = true,
        Normalization = Normalization.OneToOne,
        OutputNormalization = Normalization.MinMaxOneToOne,
        ExecutionProvider = Provider.GetProvider(),
        Path = @"M:\Extractor\Depth.onnx"
    });

    // 2. Extract Depth map from Image
    var inputImage = new ImageInput("Input.png");
    var depthMapImage = await pipeline.RunAsync(new ExtractorImageOptions
    {
        Image = inputImage
    });
    await depthMapImage.SaveAsync("Output.png");

    // 3. Extract Depth map from Video (Streaming mode)
    var inputStream = await VideoInputStream.CreateAsync("Input.mp4");
    var depthMapVideo = pipeline.RunAsync(new ExtractorStreamOptions
    {
            Stream = inputStream.GetAsync()
    });
    await depthMapVideo.SaveAync("Output.mp4");

    // 4. Add audio from the source video (optional)
    await AudioManager.AddAudioAsync("Output.mp4", "Input.mp4");
}
```

## Creating an Extractor Pipeline

```csharp
[nuget: TensorStack.Extractors]
[nuget: TensorStack.Providers.DML]

// Create the pipeline
var pipeline = ExtractorPipeline.Create(new ExtractorConfig
{
    Normalization = Normalization.ZeroToOne,
    ExecutionProvider = Provider.GetProvider(),
    Path = @"M:\Models\RealESR-General-4x\model.onnx"
});
```

**Configuration Options:**

- `Normalization` — Input value normalization (`ZeroToOne` or `OneToOne`)  
- `ExecutionProvider` — Hardware provider (CPU, GPU, DirectML, etc.)  
- `Path` — Path to the ONNX model  

---

## Extract Image Features
```csharp
    [nuget: TensorStack.Image.Bitmap]

    // Read Image
    var inputImage = new ImageInput("Input.png");

    // Extract Image
    var output = await pipeline.RunAsync(new ExtractorImageOptions
    {
        Image = inputImage
    });

    // Write Image
    await output.SaveAsync("Output.png");
```

---

## Extract Video Features (Buffered)
Buffers all frames in memory. Suitable for short-duration videos, AI-generated content, low-resolution videos, or GIFs.
```csharp
    [nuget: TensorStack.Video.Windows]

    // Read Video
    var inputVideo = await VideoInput.CreateAsync("Input.gif");

    // Extract Video
    var outputVideo = await pipeline.RunAsync(new ExtractVideoOptions
    {  
        Video = inputVideo
    });

    // Write Video
    await outputVideo.SaveAync("Output.mp4");
```

---

## Extract Video Features (Stream)
Processes frames one-by-one for minimal memory usage. Ideal for high-resolution or long-duration videos.
```csharp
    [nuget: TensorStack.Video.Windows]

    // Read Stream
    var inputStream = await VideoInputStream.CreateAsync("Input.mp4");

    // Extract Stream
    var outputStream = pipeline.RunAsync(new ExtractStreamOptions
    {
        Stream = inputStream.GetAsync()
    });

    // Write Stream
    await outputStream.SaveAync("Output.mp4");
```


---

## Audio Support
TensorStack.Video only processes video frames, so audio will be missing from the final result.

You can use the TensorStack.Audio package to restore audio from the source video:
```csharp
    [nuget: TensorStack.Audio.Windows]

    await AudioManager.AddAudioAsync("TargetVideo.mp4", "SourceVideo.mp4");
```
---


## Tiling Support
Tiling allows images and video frames to be processed in smaller sections (tiles) instead of all at once. This helps reduce memory usage and can improve performance when working with very large images or high-resolution videos.

The `TileMode` determines how these tiles are handled:

* **None:** Processes the entire image/frame in a single pass.
* **Overlap:** Tiles have overlapping edges to avoid visible seams.
* **Blend:** Overlapping tiles are blended together for smooth transitions.
* **Clip:** Tiles are cut without blending.
* **Clip + Blend:** Combines clipping and blending for high-quality results.

Additional options include:

* **MaxTileSize:** The maximum size of each tile in pixels. Smaller tiles reduce memory usage but may take longer to process.
* **TileOverlap:** The number of overlapping pixels between tiles. More overlap can prevent visible seams and improve output quality.

Adjusting these settings allows you to balance memory usage, processing speed, and visual quality for your extractor tasks.

---

## Extractor Models

Here is a list of some known and tested models compatible with `TensorStack.Extractors`:

- [Xenova/depth-anything-large-hf](https://huggingface.co/Xenova/depth-anything-large-hf)  
- [julienkay/sentis-MiDaS](https://huggingface.co/julienkay/sentis-MiDaS)  
- [axodoxian/controlnet_onnx](https://huggingface.co/axodoxian/controlnet_onnx)  
- [briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4)  
- [TensorStack/FeatureExtractor-amuse](https://huggingface.co/TensorStack/FeatureExtractor-amuse)  

---
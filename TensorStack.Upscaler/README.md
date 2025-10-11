# TensorStack.Upscaler
High-performance ONNX image and video upscaling for AI-generated content or other media. Supports multiple models and provides both buffered and streaming video modes for flexible memory usage.

---

## Quick Start

This minimal example demonstrates how to upscale an image and a video using `TensorStack.Upscaler`.

```csharp
[nuget: TensorStack.Upscaler]
[nuget: TensorStack.Providers.DML]
[nuget: TensorStack.Image.Bitmap]
[nuget: TensorStack.Video.Windows]
[nuget: TensorStack.Audio.Windows]
[model: https://huggingface.co/TensorStack/Upscale-amuse/resolve/main/RealESR-General-4x/model.onnx]

async Task QuickStartAsync()
{
    // 1. Create the Upscale Pipeline
    var pipeline = UpscalePipeline.Create(new UpscalerConfig
    {
        ScaleFactor = 4,
        Normalization = Normalization.ZeroToOne,
        ExecutionProvider = Provider.GetProvider(),
        Path = @"M:\Models\RealESR-General-4x\model.onnx"
    });

    // 2. Upscale an Image
    var inputImage = new ImageInput("Input.png");
    var upscaledImage = await pipeline.RunAsync(new UpscaleImageOptions
    {
        Image = inputImage,
        TileMode = TileMode.None
    });
    await upscaledImage.SaveAsync("Output.png");

    // 3. Upscale a Video (Streaming mode)
    var inputStream = await VideoInputStream.CreateAsync("Input.mp4");
    var upscaledVideo = pipeline.RunAsync(new UpscaleStreamOptions
    {
        Stream = inputStream.GetAsync(),
        TileMode = TileMode.None
    });
    await upscaledVideo.SaveAsync("Output.mp4");

    // 4. Add audio from the source video (optional)
    await AudioManager.AddAudioAsync("Output.mp4", "Input.mp4");
}
```

---


## Creating an Upscale Pipeline

```csharp
[nuget: TensorStack.Upscaler]
[nuget: TensorStack.Providers.DML]

// Create the pipeline
var pipeline = UpscalePipeline.Create(new UpscalerConfig
{
    ScaleFactor = 4,
    Normalization = Normalization.ZeroToOne,
    ExecutionProvider = Provider.GetProvider(),
    Path = @"M:\Models\RealESR-General-4x\model.onnx"
});
```

**Configuration Options:**

- `ScaleFactor` — Upscale factor (e.g., 2x, 4x)  
- `Normalization` — Input value normalization (`ZeroToOne` or `OneToOne`)  
- `ExecutionProvider` — Hardware provider (CPU, GPU, DirectML, etc.)  
- `Path` — Path to the ONNX model  

---



## Upscale Image
```csharp
    [nuget: TensorStack.Image.Bitmap]

    // Read Image
    var inputImage = new ImageInput("Input.png");

    // Upscale Image
    var output = await pipeline.RunAsync(new UpscaleImageOptions
    {
        Image = inputImage,
        TileMode = TileMode.None
    });

    // Write Image
    await output.SaveAsync("Output.png");
```

---


## Upscale Video (Buffered)
Buffers all frames in memory. Suitable for short-duration videos, AI-generated content, low-resolution videos, or GIFs.
```csharp
    [nuget: TensorStack.Video.Windows]

    // Read Video
    var inputVideo = await VideoInput.CreateAsync("Input.gif");

    // Upscale Video
    var outputVideo = await pipeline.RunAsync(new UpscaleVideoOptions
    {  
        Video = inputVideo,
        TileMode = TileMode.None
    });

    // Write Video
    await outputVideo.SaveAync("Output.mp4");
```

---

## Upscale Video (Stream)
Processes frames one-by-one for minimal memory usage. Ideal for high-resolution or long-duration videos.
```csharp
    [nuget: TensorStack.Video.Windows]

    // Read Stream
    var inputStream = await VideoInputStream.CreateAsync("Input.mp4");

    // Upscale Stream
    var outputStream = pipeline.RunAsync(new UpscaleStreamOptions
    {
        Stream = inputStream.GetAsync(),
        TileMode = TileMode.None
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

Adjusting these settings allows you to balance memory usage, processing speed, and visual quality for your upscaling tasks.

---

## Upscale Models

Here is a list of some known and tested models compatible with `TensorStack.Upscaler`:

- [wuminghao/swinir](https://huggingface.co/wuminghao/swinir)  
- [rocca/swin-ir-onnx](https://huggingface.co/rocca/swin-ir-onnx)  
- [Xenova/swin2SR-classical-sr-x2-64](https://huggingface.co/Xenova/swin2SR-classical-sr-x2-64)  
- [Xenova/swin2SR-classical-sr-x4-64](https://huggingface.co/Xenova/swin2SR-classical-sr-x4-64)  
- [Neus/GFPGANv1.4](https://huggingface.co/Neus/GFPGANv1.4)  
- [TensorStack/Upscale-amuse](https://huggingface.co/TensorStack/Upscale-amuse)  
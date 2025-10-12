# TensorStack.Video
`TensorStack.Video` provides the core, cross-platform abstractions for video processing in TensorStack.  
It defines shared base classes used by platform-specific implementations such as **TensorStack.Video.Windows** and **TensorStack.Video.Linux**. 

---

## Frame Interpolation
The Interpolation Pipeline uses **RIFE (Real-Time Intermediate Flow Estimation)**
RIFE analyzes motion between consecutive frames and predicts new intermediate frames, producing smoother motion and higher frame rates without traditional frame blending artifacts.  
It’s designed for both speed and quality, making it ideal for upscaling or enhancing AI-generated and low-FPS video content.

## Quick Start

This minimal example demonstrates how to perform **video frame interpolation** using `TensorStack.Video.Windows`.

```csharp
[nuget: TensorStack.Video.Windows]
[nuget: TensorStack.Providers.DML]

async Task QuickStartAsync()
{
    var provider = Provider.GetProvider();

    // Create the interpolation pipeline
    using (var pipeline = InterpolationPipeline.Create(provider))
    {
        // Read video stream
        var inputStream = new VideoInputStream("Input.mp4");

        // Interpolate the stream (e.g., 3x frame rate)
        var outputStream = pipeline.RunAsync(new InterpolationStreamOptions
        {
            Multiplier = 3,
            Stream = inputStream.GetAsync()
        });

        // Save the output video
        await outputStream.SaveAync("Output.mp4");
    }
}
```

---


- **`Multiplier`** — Defines how many new frames are generated between existing ones.  
  For example, a value of `3` triples the frame rate (turning 30 FPS into 90 FPS).  
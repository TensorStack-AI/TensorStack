# TensorStack.Video.Windows
`TensorStack.Video.Windows` provides Windows-specific support for reading and writing video.  
It facilitates efficient access to video streams, enabling frames to be read, processed, and saved back to common formats using the Windows media stack.

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
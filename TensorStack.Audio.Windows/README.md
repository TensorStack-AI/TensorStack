# TensorStack.Audio.Windows
`TensorStack.Audio.Windows` provides Windows-specific support for reading and writing audio using FFmpeg. It allows loading audio information, reading and writing audio tensors, and muxing audio into video files.

---

## AudioInput
`AudioInput` wraps an audio file into a convenient object backed by an `AudioTensor`. It allows loading, manipulating, and saving audio.

```csharp
// Load Audio

AudioInput audioInput = new AudioInput("speech.wav");

AudioInput audioInputAsync = await AudioInput.CreateAsync("speech.wav");


// With custom codec, sample rate, and channels

AudioInput audioInput = new AudioInput("music.mp3", "pcm_f32le", 44100, 2);

AudioInput audioInputAsync = await AudioInput.CreateAsync("music.mp3", "pcm_f32le", 44100, 2);


// Save Audio

audioInput.Save("output.wav");

await audioInput.SaveAsync("output.wav");
```



## Notes

* `AudioInput` uses `AudioManager` internally to handle the audio tensor.
* `AudioManager.Initialize` only needed for custom FFmpeg/FFprobe binaries or a different temp directory; the NuGet package provides defaults.

---

# AudioManager
`AudioManager` is a static helper class for loading, saving, and processing audio files. It can read audio into tensors, write tensors back to audio files, extract audio from videos, and add audio to videos. It uses FFmpeg/FFprobe under the hood and provides both synchronous and asynchronous methods.

## Load Audio Information
```csharp
AudioInfo info = AudioManager.LoadInfo("file.wav");

AudioInfo info = await AudioManager.LoadInfoAsync("file.wav");
```

Returns metadata including codec, sample rate, channels, duration, and sample count.

---

## Load Audio Tensor
```csharp
AudioTensor tensor = AudioManager.LoadTensor("file.wav", "pcm_s16le", 16000, 1);

AudioTensor tensor = await AudioManager.LoadTensorAsync("file.wav", "pcm_s16le", 16000, 1);
```

`AudioTensor` contains the raw audio samples in float32 format.

---

## Save Audio Tensor
```csharp
AudioManager.SaveAudio("output.wav", tensor);

await AudioManager.SaveAudioAync("output.wav", tensor);
```

---

## Add Audio to Video
```csharp
AudioManager.AddAudio("video.mp4", "sourceAudio.mp3");

await AudioManager.AddAudioAsync("video.mp4", "sourceAudio.mp3");
```

This muxes the audio from the source file into the target video.

---
## Initialization

The NuGet package supplies FFmpeg binaries. Initialization is only needed if you want to use custom binaries or a different location:

```csharp
AudioManager.Initialize("ffmpeg.exe", "ffprobe.exe", "Temp");
```

This sets up the executable paths and temporary directory used for conversions.

---
## Notes

* All audio I/O uses FFmpeg under the hood.
* Asynchronous methods use `Task` and support cancellation.
* Audio data is handled in `float32` format internally.

---

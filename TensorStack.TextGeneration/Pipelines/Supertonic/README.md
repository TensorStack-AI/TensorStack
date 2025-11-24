# Supertonic TTS
https://github.com/supertone-inc/supertonic


```csharp
// [model] https://huggingface.co/TensorStack/Supertonic-onnx

var provider = Provider.GetProvider(GraphOptimizationLevel.ORT_ENABLE_ALL);
var modelPath = "M:\\Models\\Supertonic-onnx";
var pipeline = SupertonicPipeline.Create(modelPath, provider);
var options = new SupertonicOptions
{
    TextInput = "On a quiet morning in the old town, a clockmaker named Ellis unlocked his tiny shop",
    VoiceStyle = "Female1"
};

var generateResult = await pipeline.RunAsync(options);
AudioManager.SaveAudio("Output.wav", generateResult);
```

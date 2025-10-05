# Whisper

### Greedy
```csharp
var provider = Provider.GetProvider(0);
var modelPath = "M:\\Models\\Whisper-Base";
var pipeline = WhisperPipeline.Create(provider, modelPath, WhisperType.Base);
var options = new GenerateOptions
{
    Task = TaskType.Transcribe,
    Language = LanguageType.EN,
    AudioInput = await AudioInput.CreateAsync("kennedy.mp3")
};

var generateResult = await pipeline.RunAsync(options);
System.Console.WriteLine(generateResult.Result);
```

### Beam Search
```csharp
var provider = Provider.GetProvider(0);
var modelPath = "M:\\Models\\Whisper-Large";
var pipeline = WhisperPipeline.Create(provider, modelPath, WhisperType.Large);
var options = new SearchOptions
{
    Seed = 0,
    TopK = 50,
    Beams = 3,
    TopP = 0.9f,
    Temperature = 1f,
    LengthPenalty = -1f,
    DiversityLength = 20,
    NoRepeatNgramSize = 3,
    EarlyStopping = EarlyStopping.BestBeam,
    Task = TaskType.Transcribe,
    Language = LanguageType.EN,
    AudioInput = await AudioInput.CreateAsync("kennedy.mp3")
};

foreach (var beamResult in await pipeline.RunAsync(options))
{
    System.Console.WriteLine(beamResult.Result);
}
```

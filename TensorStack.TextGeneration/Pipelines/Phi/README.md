# Phi3 Pipeline

### Greedy
```csharp
var provider = Provider.GetProvider(0);
var modelPath = "M:\\Models\\Phi-3-medium-128k-instruct-onnx-directml";
var pipeline = Phi3Pipeline.Create(provider, modelPath, PhiType.Mini);
var options = new GenerateOptions
{
    Prompt = "<|user|>What is an apple?<|end|><|assistant|>"
};

var generateResult = await pipeline.RunAsync(options);
System.Console.WriteLine(generateResult.Result);
```

### Beam Search
```csharp
var provider = Provider.GetProvider(0);
var modelPath = "M:\\Models\\Phi-3-medium-128k-instruct-onnx-directml";
var pipeline = Phi3Pipeline.Create(provider, modelPath, PhiType.Mini);
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
    EarlyStopping = EarlyStopping.None,
    Prompt = "<|user|>What is an apple?<|end|><|assistant|>"
};

foreach (var beamResult in await pipeline.RunAsync(options))
{
    System.Console.WriteLine(beamResult.Result);
}
```
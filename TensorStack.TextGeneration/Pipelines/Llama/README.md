# Llama Pipeline

Llama-3.2-1B: https://huggingface.co/TensorStack/Llama-3.2-1B

### Greedy
```csharp
var provider = Provider.GetProvider();
var modelPath = "M:\\Models\\Llama-3.2-1B";
var pipeline = LlamaPipeline.Create(provider, modelPath, PhiType.Mini);
var options = new GenerateOptions
{
    Prompt = "What is an apple?"
};

var generateResult = await pipeline.RunAsync(options);
System.Console.WriteLine(generateResult.Result);
```

### Beam Search
```csharp
var provider = Provider.GetProvider();
var modelPath = "M:\\Models\\Llama-3.2-1B";
var pipeline = LlamaPipeline.Create(provider, modelPath, PhiType.Mini);
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
    Prompt = "What is an apple?"
};

foreach (var beamResult in await pipeline.RunAsync(options))
{
    System.Console.WriteLine(beamResult.Result);
}
```
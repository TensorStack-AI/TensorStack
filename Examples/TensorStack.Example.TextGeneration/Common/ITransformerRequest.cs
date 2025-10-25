using TensorStack.TextGeneration.Common;

namespace TensorStack.Example.Common
{
    public interface ITransformerRequest
    {
        int Beams { get; set; }
        int DiversityLength { get; set; }
        EarlyStopping EarlyStopping { get; set; }
        float LengthPenalty { get; set; }
        int MaxLength { get; set; }
        int MinLength { get; set; }
        int NoRepeatNgramSize { get; set; }
        string Prompt { get; set; }
        int Seed { get; set; }
        float Temperature { get; set; }
        int TopK { get; set; }
        float TopP { get; set; }
    }
}
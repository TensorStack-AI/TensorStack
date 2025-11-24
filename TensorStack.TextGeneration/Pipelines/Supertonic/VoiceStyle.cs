using TensorStack.Common.Tensor;

namespace TensorStack.TextGeneration.Pipelines.Supertonic
{
    public record VoiceStyle
    {
        public VoiceStyle(string name, Tensor<float> global, Tensor<float> dropout)
        {
            Name = name;
            Global = global;
            Dropout = dropout;
        }

        public string Name { get; init; }
        public Tensor<float> Global { get; init; }
        public Tensor<float> Dropout { get; init; }
    }
}

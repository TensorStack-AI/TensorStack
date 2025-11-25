using TensorStack.Common.Pipeline;

namespace TensorStack.TextGeneration.Pipelines.Supertonic
{
    public record SupertonicOptions : IRunOptions
    {
        public string TextInput { get; set; }
        public string VoiceStyle { get; set; }
        public int Steps { get; set; } = 5;
        public float Speed { get; set; } = 1f;
        public float SilenceDuration { get; set; } = 0.3f;
        public int Seed { get; set; }
    }
}

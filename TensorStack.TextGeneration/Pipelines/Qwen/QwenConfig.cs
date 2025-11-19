using TensorStack.TextGeneration.Common;

namespace TensorStack.TextGeneration.Pipelines.Qwen
{
    public record QwenConfig : TransformerConfig
    {
        public bool OutputLastHiddenStates { get; set; }
    }
}

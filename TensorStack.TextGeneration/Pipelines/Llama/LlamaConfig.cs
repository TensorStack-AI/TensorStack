using TensorStack.TextGeneration.Common;

namespace TensorStack.TextGeneration.Pipelines.Llama
{
    public record LlamaConfig : TransformerConfig
    {
        public bool OutputLastHiddenStates { get; set; }
    }
}

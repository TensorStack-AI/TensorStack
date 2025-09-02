using TensorStack.Common;

namespace TensorStack.TextGeneration.Common
{
    public record DecoderConfig : ModelConfig
    {
        public int VocabSize { get; set; }
        public int NumHeads { get; set; }
        public int NumLayers { get; set; }
        public int HiddenSize { get; set; }
        public int NumKVHeads { get; set; }
    }
}

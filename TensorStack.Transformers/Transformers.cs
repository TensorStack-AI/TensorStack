using TensorStack.Common;
using TensorStack.Transformers.Tokenizers;

namespace TensorStack.Transformers
{

    public record DecoderConfig : ModelConfig
    {
        public int VocabSize { get; set; }
        public int NumHeads { get; set; }
        public int NumLayers { get; set; }
        public int HiddenSize { get; set; }
    }

    public record EncoderConfig : ModelConfig
    {
        public int VocabSize { get; set; }
        public int NumHeads { get; set; }
        public int NumLayers { get; set; }
        public int HiddenSize { get; set; }
    }


    public record TransformerConfig
    {
        public T5TokenizerConfig TokenizerConfig { get; set; }
        public EncoderConfig EncoderConfig { get; set; }
        public DecoderConfig DecoderConfig { get; set; }
    }


    public record Phi3Config
    {
        public T5TokenizerConfig TokenizerConfig { get; set; }
        public DecoderConfig DecoderConfig { get; set; }
    }
}

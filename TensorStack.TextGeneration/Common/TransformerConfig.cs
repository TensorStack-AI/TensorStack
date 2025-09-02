using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.TextGeneration.Common
{
    public abstract record TransformerConfig
    {
        public ITokenizer Tokenizer { get; set; }
        public EncoderConfig EncoderConfig { get; set; }
        public DecoderConfig DecoderConfig { get; set; }
    }
}

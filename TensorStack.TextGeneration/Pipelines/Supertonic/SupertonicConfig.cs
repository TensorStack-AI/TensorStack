using TensorStack.Common;

namespace TensorStack.TextGeneration.Pipelines.Supertonic
{
    public record SupertonicConfig
    {
        public int SampleRate { get; init; } = 44100;
        public int BaseChunkSize { get; init; } = 512;
        public int LatentDim { get; init; } = 24;
        public int ChunkCompressFactor { get; init; } = 6;
        public int TextEmbedSize { get; init; } = 256;
        public int ScaleFactor { get; init; } = 3072;
        public string IndexerPath { get; init; }
        public string VoiceStylePath { get; init; }
        public ModelConfig PredictorConfig { get; init; }
        public ModelConfig EncoderConfig { get; init; }
        public ModelConfig EstimatorConfig { get; init; }
        public ModelConfig DecoderConfig { get; init; }
    }
}

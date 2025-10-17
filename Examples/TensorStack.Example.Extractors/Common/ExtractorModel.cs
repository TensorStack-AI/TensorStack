using TensorStack.Common;

namespace TensorStack.Example.Common
{
    public record ExtractorModel
    {
        public int Id { get; init; }
        public string Name { get; init; }
        public bool IsDefault { get; set; }

        public int Channels { get; set; }
        public int SampleSize { get; set; }
        public Normalization Normalization { get; set; }
        public Normalization OutputNormalization { get; set; }
        public int OutputChannels { get; set; }
        public bool IsDynamicOutput { get; set; }
        public string Path { get; set; }
        public string UrlPath { get; set; }
    }
}

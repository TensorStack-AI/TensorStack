using TensorStack.Common;
using TensorStack.WPF;

namespace TensorStack.Example.Common
{

    public class UpscaleModel : BaseModel
    {
        public int Id { get; init; }
        public string Name { get; init; }
        public bool IsDefault { get; set; }

        public int Channels { get; init; } = 3;
        public int SampleSize { get; init; }
        public int ScaleFactor { get; init; } = 1;
        public Normalization Normalization { get; init; }
        public string Path { get; set; }
        public string UrlPath { get; set; }
    }
}

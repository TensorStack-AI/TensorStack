using TensorStack.Common;
using TensorStack.WPF;

namespace TensorStack.Example.Common
{

    public class BackgroundModel : BaseModel
    {
        public int Id { get; init; }
        public string Name { get; init; }
        public bool IsDefault { get; set; }


        public int Channels { get; init; } = 3;
        public int SampleSize { get; init; }
        public Normalization Normalization { get; init; } = Normalization.ZeroToOne;
        public Normalization OutputNormalization { get; init; }
        public int OutputChannels { get; init; } = 1;
        public string Path { get; set; }
    }
}

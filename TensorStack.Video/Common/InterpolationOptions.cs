using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.Common.Video;

namespace TensorStack.Video.Common
{
    /// <summary>
    /// Base Interpolation Options.
    /// </summary>
    public abstract record InterpolationOptions : IRunOptions
    {
        public int Multiplier { get; init; }
    }



    /// <summary>
    /// Video Interpolation Options.
    /// </summary>
    public sealed record InterpolationVideoOptions : InterpolationOptions
    {
        public VideoTensor Video { get; init; }
    }



    /// <summary>
    /// Stream Interpolation Options.
    /// </summary>
    public sealed record InterpolationStreamOptions : InterpolationOptions
    {
        public VideoStream Stream { get; init; }
    }
}

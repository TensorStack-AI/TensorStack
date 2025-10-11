using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common.Tensor;
using TensorStack.Common.Video;

namespace TensorStack.Video
{
    public static class Extensions
    {
        public static Task SaveAync(this VideoTensor videoTensor, string videoFile, string videoCodec = "mp4v", float? frameRateOverride = default, CancellationToken cancellationToken = default)
        {
            return VideoManager.SaveVideoTensorAync(videoFile, videoTensor, videoCodec, frameRateOverride, cancellationToken: cancellationToken);
        }

        public static Task SaveAync(this IAsyncEnumerable<VideoFrame> videoFrames, string videoFile, string videoCodec = "mp4v", int? widthOverride = null, int? heightOverride = null, float? frameRateOverride = null, CancellationToken cancellationToken = default)
        {
            return VideoManager.WriteVideoStreamAsync(videoFile, videoFrames, videoCodec, widthOverride, heightOverride, frameRateOverride, cancellationToken);
        }
    }
}

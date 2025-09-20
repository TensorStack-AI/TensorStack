using System;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common.Video;

namespace TensorStack.Video
{
    public interface IVideoService
    {
        Task<VideoInfo> GetInfoAsync(string filename);
        Task<VideoInputStream> GetStreamAsync(string filename);
        Task<VideoInputStream> SaveStreamAync(VideoInputStream videoInput, string videoOutputFile, Func<VideoFrame, Task<VideoFrame>> frameProcessor, CancellationToken cancellationToken = default);
    }
}

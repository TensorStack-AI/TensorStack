// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Threading.Tasks;

namespace TensorStack.Video
{
    public interface IVideoService
    {
        Task<VideoInfo> GetVideoInfoAsync(string filename);
    }
}

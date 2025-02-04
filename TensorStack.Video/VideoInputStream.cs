// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.Common.Video;

namespace TensorStack.Video
{
    public class VideoInputStream
    {
        private readonly string _filename;
        private readonly string _videoCodec;
        private readonly VideoInfo _videoInfo;

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoInputStream"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="videoCodec">The video codec.</param>
        /// <exception cref="System.Exception">Failed to open video file.</exception>
        public VideoInputStream(string filename, string videoCodec = "mp4v")
        {
            _filename = filename;
            _videoCodec = videoCodec;
            _videoInfo = VideoService.LoadVideoInfo(_filename);
        }

        /// <summary>
        /// Gets the filename.
        /// </summary>
        /// <value>The filename.</value>
        public string Filename => _filename;

        /// <summary>
        /// Gets the video width.
        /// </summary>
        public int Width => _videoInfo.Width;

        /// <summary>
        /// Gets the video height.
        /// </summary>
        public int Height => _videoInfo.Height;

        /// <summary>
        /// Gets the video frame rate.
        /// </summary>
        public float FrameRate => _videoInfo.FrameRate;

        /// <summary>
        /// Gets the video frame count.
        /// </summary>
        public int FrameCount => _videoInfo.FrameCount;


        /// <summary>
        /// Gets the VideoFrame stream.
        /// </summary>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>IAsyncEnumerable&lt;ImageFrame&gt;.</returns>
        public IAsyncEnumerable<VideoFrame> GetAsync(int? width = default, int? height = default, float? frameRate = default, ResizeMode resizeMode = ResizeMode.Stretch, CancellationToken cancellationToken = default)
        {
            return VideoService.ReadStreamAsync(_filename, frameRate, width, height, resizeMode, cancellationToken);
        }


        /// <summary>
        /// Saves the VideoFrame stream.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="stream">The stream.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>Task.</returns>
        public Task SaveAsync(IAsyncEnumerable<VideoFrame> stream, string filename, int? width = default, int? height = default, float? frameRate = default, CancellationToken cancellationToken = default)
        {
            return VideoService.WriteVideoStreamAsync(stream, filename, _videoCodec, frameRate, width, height, cancellationToken);
        }


        /// <summary>
        /// Create a buffered VideoTensor of this stream.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;VideoTensor&gt; representing the asynchronous operation.</returns>
        public async Task<VideoTensor> CreateTensorAsync(int? width = default, int? height = default, float? frameRate = default, ResizeMode resizeMode = ResizeMode.Stretch, CancellationToken cancellationToken = default)
        {
            var buffered = await GetAsync(width, height, frameRate, resizeMode, cancellationToken)
                .Select(x => x.Frame)
                .ToArrayAsync(cancellationToken);
            return new VideoTensor(buffered.Join(), frameRate ?? FrameRate);
        }
    }
}

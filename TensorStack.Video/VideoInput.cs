// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common.Tensor;
using TensorStack.Common.Video;

namespace TensorStack.Video
{
    /// <summary>
    /// Class to handle processing of a video stream.
    /// </summary>
    public class VideoInput
    {
        private readonly string _filename;
        private readonly int _width;
        private readonly int _height;
        private readonly float _frameRate;
        private readonly int _frameCount;
        private readonly string _videoCodec;

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoInput"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="videoCodec">The video codec.</param>
        /// <exception cref="System.Exception">Failed to open video file.</exception>
        public VideoInput(string filename, string videoCodec = "mp4v")
        {
            _filename = filename;
            _videoCodec = videoCodec;
            using (var videoReader = new VideoCapture(_filename))
            {
                if (!videoReader.IsOpened())
                    throw new Exception("Failed to open video file.");

                _width = videoReader.FrameWidth;
                _height = videoReader.FrameHeight;
                _frameRate = (float)videoReader.Fps;
                _frameCount = videoReader.FrameCount;
            }
        }

        /// <summary>
        /// Gets the filename.
        /// </summary>
        /// <value>The filename.</value>
        public string Filename => _filename;

        /// <summary>
        /// Gets the video width.
        /// </summary>
        public int Width => _width;

        /// <summary>
        /// Gets the video height.
        /// </summary>
        public int Height => _height;

        /// <summary>
        /// Gets the video frame rate.
        /// </summary>
        public float FrameRate => _frameRate;

        /// <summary>
        /// Gets the video frame count.
        /// </summary>
        public int FrameCount => _frameCount;


        /// <summary>
        /// Gets a memory buffered VideoTensor
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>Task&lt;VideoTensor&gt;.</returns>
        public Task<VideoTensor> GetTensorAsync(int? width = default, int? height = default, float? frameRate = default, CancellationToken cancellationToken = default)
        {
            return Extensions.GetVideoTensorAsync(this, frameRate, width, height, cancellationToken);
        }


        /// <summary>
        /// Gets the VideoFrame stream.
        /// </summary>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>IAsyncEnumerable&lt;ImageFrame&gt;.</returns>
        public IAsyncEnumerable<VideoFrame> GetStreamAsync(int? width = default, int? height = default, float? frameRate = default, CancellationToken cancellationToken = default)
        {
            return Extensions.ReadVideoFramesAsync(_filename, frameRate, width, height, cancellationToken);
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
        public Task SaveStreamAsync(IAsyncEnumerable<VideoFrame> stream, string filename, int? width = default, int? height = default, float? frameRate = default, CancellationToken cancellationToken = default)
        {
            return Extensions.WriteVideoStreamAsync(stream, filename, _videoCodec, frameRate, width, height, cancellationToken);
        }

    }
}

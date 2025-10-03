// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using TensorStack.Common.Tensor;

namespace TensorStack.Video
{
    public class VideoInputStreamBase
    {
        private readonly VideoInfo _videoInfo;

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoInputStreamBase"/> class.
        /// </summary>
        /// <param name="videoInfo">The video information.</param>
        /// <param name="videoCodec">The video codec.</param>
        public VideoInputStreamBase(VideoInfo videoInfo)
        {
            _videoInfo = videoInfo;
        }

        /// <summary>
        /// Gets the filename.
        /// </summary>
        /// <value>The filename.</value>
        public string SourceFile => _videoInfo.FileName;

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
        /// Gets the video codec.
        /// </summary>
        public string VideoCodec => _videoInfo.VideoCodec;

        /// <summary>
        /// Gets the duration.
        /// </summary>
        public TimeSpan Duration => _videoInfo.Duration;

        /// <summary>
        /// Gets the thumbnail.
        /// </summary>
        public ImageTensor Thumbnail => _videoInfo.Thumbnail;

    }
}

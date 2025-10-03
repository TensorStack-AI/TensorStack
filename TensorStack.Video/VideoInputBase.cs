// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common.Tensor;

namespace TensorStack.Video
{
    /// <summary>
    /// Class to handle processing of a video.
    /// </summary>
    public abstract class VideoInputBase : VideoTensor
    {
           /// <summary>
        /// Initializes a new instance of the <see cref="VideoInput"/> class.
        /// </summary>
        /// <param name="videoTensor">The video tensor.</param>
        public VideoInputBase(VideoTensor videoTensor) : base(videoTensor, videoTensor.FrameRate) { }


        /// <summary>
        /// Gets the source video filename.
        /// </summary>
        public abstract string SourceFile { get; }


        /// <summary>
        /// Save the Video to file
        /// </summary>
        /// <param name="filename">The filename.</param>
        public abstract void Save(string filename);

        /// <summary>
        /// Save the Video to file
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="cancellationToken">The cancellation token</param>
        public abstract Task SaveAsync(string filename, CancellationToken cancellationToken = default);
    }
}

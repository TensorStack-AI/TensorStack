// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;

namespace TensorStack.Video
{
    /// <summary>
    /// Class to handle processing of a video stream.
    /// </summary>
    public class VideoInput : VideoInputBase
    {
        private readonly string _sourceFile;

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoInput"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="videoCodec">The video codec.</param>
        public VideoInput(string filename, int? widthOverride = default, int? heightOverride = default, float? frameRateOverride = default, ResizeMode resizeMode = ResizeMode.Crop)
            : this(filename, VideoManager.LoadVideoTensor(filename, widthOverride, heightOverride, frameRateOverride, resizeMode)) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoInput"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="videoTensor">The video tensor.</param>
        public VideoInput(string filename, VideoTensor videoTensor)
            : base(videoTensor)
        {
            _sourceFile = filename;
        }


        /// <summary>
        /// Gets the source video filename.
        /// </summary>
        public override string SourceFile => _sourceFile;


        /// <summary>
        /// Save the Video to file
        /// </summary>
        /// <param name="filename">The filename.</param>
        public override void Save(string filename)
        {
            throw new System.NotImplementedException();
        }


        /// <summary>
        /// Save the Video to file
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="cancellationToken">The cancellation token</param>
        public override Task SaveAsync(string filename, CancellationToken cancellationToken = default)
        {
            return SaveAsync(filename, frameRateOverride: default, cancellationToken: cancellationToken);
        }

        /// <summary>
        /// Save the VideoTensor to file
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="framerateOverride">The framerate.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        public async Task SaveAsync(string filename, string videoCodec = "mp4v", float? frameRateOverride = default, CancellationToken cancellationToken = default)
        {
            await VideoManager.SaveVideoTensorAync(filename, this, videoCodec, frameRateOverride, cancellationToken);
        }


        /// <summary>
        /// Create a VideoInput asynchronously
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="widthOverride">The width.</param>
        /// <param name="heightOverride">The height.</param>
        /// <param name="frameRateOverride ">The frame rate.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;VideoInput&gt; representing the asynchronous operation.</returns>
        public static async Task<VideoInput> CreateAsync(string filename, int? widthOverride = default, int? heightOverride = default, float? frameRateOverride = default, ResizeMode resizeMode = ResizeMode.Crop, CancellationToken cancellationToken = default)
        {
            return new VideoInput(filename, await VideoManager.LoadVideoTensorAsync(filename, widthOverride, heightOverride, frameRateOverride, resizeMode, cancellationToken));
        }

    }
}

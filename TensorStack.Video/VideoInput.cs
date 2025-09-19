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
    public class VideoInput : VideoTensor
    {
        private string _filename;

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoInput"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="videoCodec">The video codec.</param>
        public VideoInput(string filename, int? widthOverride = default, int? heightOverride = default, float? frameRateOverride = default, ResizeMode resizeMode = ResizeMode.Crop)
            : this(VideoService.LoadVideoTensor(filename, widthOverride, heightOverride, frameRateOverride, resizeMode))
        {
            _filename = filename;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoInput"/> class.
        /// </summary>
        /// <param name="videoTensor">The video tensor.</param>
        public VideoInput(VideoTensor videoTensor) : base(videoTensor, videoTensor.FrameRate) { }

        /// <summary>
        /// Gets the filename.
        /// </summary>
        /// <value>The filename.</value>
        public string Filename => _filename;


        /// <summary>
        /// Save the VideoTensor to file
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="framerateOverride">The framerate.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        public async Task SaveAsync(string filename, string videoCodec = "mp4v", float? frameRateOverride = default, CancellationToken cancellationToken = default)
        {
            await VideoService.SaveVideoTensorAync(filename, this, videoCodec, frameRateOverride, cancellationToken);
        }


        /// <summary>
        /// Creates the stream.
        /// </summary>
        /// <returns>VideoStream.</returns>
        public VideoInputStream CreateStream()
        {
            return new VideoInputStream(_filename);
        }


        /// <summary>
        /// Creates the stream asynchronously.
        /// </summary>
        /// <returns>Task&lt;VideoInputStream&gt;.</returns>
        public Task<VideoInputStream> CreateStreamAsync()
        {
            return VideoInputStream.CreateAsync(_filename);
        }


        /// <summary>
        /// Sets the filename.
        /// </summary>
        /// <param name="filename">The filename.</param>
        public void SetFilename(string filename)
        {
            _filename = filename;
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
            return new VideoInput(await VideoService.LoadVideoTensorAsync(filename, widthOverride, heightOverride, frameRateOverride, resizeMode, cancellationToken));
        }
    }
}

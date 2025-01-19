// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using TensorStack.Common.Tensor;

namespace TensorStack.Common.Video
{
    /// <summary>
    /// VideoFrame for streaming functions.
    /// </summary>
    public class VideoFrame
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="VideoFrame"/> class.
        /// </summary>
        /// <param name="frame">The frame.</param>
        /// <param name="sourceFrameRate">The source frame rate.</param>
        public VideoFrame(ImageTensor frame, float sourceFrameRate, ImageTensor auxFrame = default)
        {
            Frame = frame;
            AuxFrame = auxFrame;
            SourceFrameRate = sourceFrameRate;
            ThrowIfInvalid();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoFrame"/> class.
        /// </summary>
        /// <param name="frame">The frame.</param>
        /// <param name="sourceFrameRate">The source frame rate.</param>
        public VideoFrame(Tensor<float> frame, float sourceFrameRate)
            : this(new ImageTensor(frame), sourceFrameRate) { }


        /// <summary>
        /// Gets the frame.
        /// </summary>
        public ImageTensor Frame { get; }

        /// <summary>
        /// Gets the source frame rate.
        /// </summary>
        public float SourceFrameRate { get; }

        /// <summary>
        /// Gets the channel count.
        /// </summary>
        public int Channels => Frame.Dimensions[1];

        /// <summary>
        /// Gets the frame height.
        /// </summary>
        public int Height => Frame.Dimensions[2];

        /// <summary>
        /// Gets the frame width.
        /// </summary>
        public int Width => Frame.Dimensions[3];

        /// <summary>
        /// Gets or sets the control frame.
        /// </summary>
        public ImageTensor AuxFrame { get; set; }

        /// <summary>
        /// Throws if VideoFrame is invalid.
        /// </summary>
        /// <exception cref="System.ArgumentNullException">Frame</exception>
        protected void ThrowIfInvalid()
        {
            ArgumentNullException.ThrowIfNull(Frame, nameof(Frame));
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(SourceFrameRate, 0, nameof(SourceFrameRate));
            if (AuxFrame != null && (Frame.Width != AuxFrame.Width || Frame.Height != AuxFrame.Height))
                throw new ArgumentException($"Frame({Frame.Width}x{Frame.Height}) and AuxFrame({AuxFrame.Width}x{AuxFrame.Height}) must have same Width/Height");
        }

    }
}

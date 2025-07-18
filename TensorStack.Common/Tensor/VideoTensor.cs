﻿// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;

namespace TensorStack.Common.Tensor
{
    /// <summary>
    /// Class to handle video in Tensor format
    /// Implements the <see cref="Tensor{float}" />
    /// </summary>
    /// <seealso cref="Tensor{float}" />
    public class VideoTensor : Tensor<float>
    {
        protected float _frameRate;
        protected TimeSpan _duration;

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoTensor"/> class.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="frameRate">The source video frame rate.</param>
        public VideoTensor(Tensor<float> tensor, float frameRate)
            : base(tensor.Memory, tensor.Dimensions)
        {
            _frameRate = frameRate;
            _duration = TimeSpan.FromSeconds(Frames / FrameRate);
            ThrowIfInvalid();
        }

        /// <summary>
        /// Gets the frame rate.
        /// </summary>
        public float FrameRate => _frameRate;

        /// <summary>
        /// Gets the number of video frames.
        /// </summary>
        public int Frames => Dimensions[0];

        /// <summary>
        /// Gets the video channel count (RGB, RGBA etc).
        /// </summary>
        public int Channels => Dimensions[1];

        /// <summary>
        /// Gets the video height.
        /// </summary>
        public int Height => Dimensions[2];

        /// <summary>
        /// Gets the video width.
        /// </summary>
        public int Width => Dimensions[3];

        /// <summary>
        /// Gets the duration.
        /// </summary>
        public TimeSpan Duration => _duration;


        /// <summary>
        /// Normalizes the tensor values from range -1 to 1 to 0 to 1.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        public void NormalizeZeroToOne()
        {
            this.NormalizeZeroOne();
        }


        /// <summary>
        /// Normalizes the tensor values from range 0 to 1 to -1 to 1.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        public void NormalizeOneToOne()
        {
            this.NormalizeOneOne();
        }


        /// <summary>
        /// Gets the frames.
        /// </summary>
        /// <returns>IEnumerable&lt;ImageTensor&gt;.</returns>
        public IEnumerable<ImageTensor> GetFrames()
        {
            foreach (var imageTensor in this.Split())
            {
                yield return new ImageTensor(imageTensor);
            }
        }


        /// <summary>
        /// Called when Tensor data has changed
        /// </summary>
        protected override void OnTensorDataChanged()
        {
            _duration = TimeSpan.FromSeconds(Frames / FrameRate);
            base.OnTensorDataChanged();
        }


        /// <summary>
        /// Throws if Dimensions are invalid.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        /// <param name="framerate">The framerate.</param>
        protected void ThrowIfInvalid()
        {
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(FrameRate, 0, nameof(FrameRate));
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(Frames, 1, nameof(Frames));
            ArgumentOutOfRangeException.ThrowIfEqual(Channels, 2, nameof(Channels));
            ArgumentOutOfRangeException.ThrowIfLessThan(Channels, 1, nameof(Channels));
            ArgumentOutOfRangeException.ThrowIfGreaterThan(Channels, 4, nameof(Channels));
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(Height, 0, nameof(Height));
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(Width, 0, nameof(Width));
        }
    }
}

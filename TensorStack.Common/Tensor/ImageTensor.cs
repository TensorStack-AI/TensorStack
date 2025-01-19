// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;

namespace TensorStack.Common.Tensor
{
    /// <summary>
    /// ImageTensor to handle Tensor data as an image.
    /// Implements the <see cref="Tensor{float}" />
    /// </summary>
    /// <seealso cref="Tensor{float}" />
    public class ImageTensor : Tensor<float>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ImageTensor"/> class.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        public ImageTensor(Tensor<float> tensor)
            : base(tensor.Memory, tensor.Dimensions)
        {
            ThrowIfInvalid();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ImageTensor"/> class.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        public ImageTensor(ReadOnlySpan<int> dimensions)
            : base(dimensions)
        {
            ThrowIfInvalid();
        }

        /// <summary>
        /// Gets the channel count (RGB, RGBA etc).
        /// </summary>
        public int Channels => Dimensions[1];

        /// <summary>
        /// Gets the image height.
        /// </summary>
        public int Height => Dimensions[2];

        /// <summary>
        /// Gets the image width.
        /// </summary>
        public int Width => Dimensions[3];

        /// <summary>
        /// Normalizes the tensor values from range -1 to 1 to 0 to 1.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        public void NormalizeZeroToOne()
        {
            this.NormalizeOneOneToZeroOne();
        }


        /// <summary>
        /// Normalizes the tensor values from range 0 to 1 to -1 to 1.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        public void NormalizeOneToOne()
        {
            this.NormalizeZeroOneToOneOne();
        }


        /// <summary>
        /// Gets a TensorSpan with the specified channels. (1 = Greyscale, 3 = RGB, 4 = RGBA)
        /// </summary>
        /// <param name="count">The channels count.</param>
        /// <returns>TensorSpan&lt;System.Single&gt;.</returns>
        public TensorSpan<float> GetChannels(int channels)
        {
            if (Channels == channels)
                return new TensorSpan<float>(Memory.Span, Dimensions);

            var channelSize = Height * Width;
            var channelDimensions = new int[] { 1, channels, Height, Width };
            return new TensorSpan<float>(Memory.Span.Slice(0, channelSize * channels), channelDimensions);
        }


        /// <summary>
        /// Throws if Dimensions are invalid.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        protected void ThrowIfInvalid()
        {
            ArgumentOutOfRangeException.ThrowIfGreaterThan(Dimensions[0], 1, "Batch");
            ArgumentOutOfRangeException.ThrowIfEqual(Channels, 2, nameof(Channels));
            ArgumentOutOfRangeException.ThrowIfLessThan(Channels, 1, nameof(Channels));
            ArgumentOutOfRangeException.ThrowIfGreaterThan(Channels, 4, nameof(Channels));
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(Height, 0, nameof(Height));
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(Width, 0, nameof(Width));
        }
    }
}

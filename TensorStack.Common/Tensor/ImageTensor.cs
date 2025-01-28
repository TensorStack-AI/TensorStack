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
            ArgumentOutOfRangeException.ThrowIfGreaterThan(channels, Channels);
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(channels, 0);
            if (Channels == channels)
                return this.AsTensorSpan();

            var channelSize = Height * Width;
            var channelDimensions = new int[] { 1, channels, Height, Width };
            return new TensorSpan<float>(Memory.Span.Slice(0, channelSize * channels), channelDimensions);
        }


        /// <summary>
        /// Gets the specified channel. (1=R, 2=G, 3=B, 4=A)
        /// </summary>
        /// <param name="channel">The channel.</param>
        /// <returns>Span&lt;System.Single&gt;.</returns>
        public Span<float> GetChannel(int channel)
        {
            ArgumentOutOfRangeException.ThrowIfGreaterThan(channel, Channels);
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(channel, 0);

            var channelSize = Height * Width;
            var startIndex = channelSize * (channel - 1);
            return Memory.Span.Slice(startIndex, channelSize);
        }


        /// <summary>
        /// Updates the channel. (1=R, 2=G, 3=B, 4=A)
        /// </summary>
        /// <param name="channel">The channel.</param>
        /// <param name="channelData">The channel data.</param>
        public void UpdateChannel(int channel, ReadOnlySpan<float> channelData)
        {
            var channelSpan = GetChannel(channel);
            for (int i = 0; i < channelSpan.Length; i++)
            {
                channelSpan[i] = channelData[i];
            }
            OnTensorDataChanged();
        }


        /// <summary>
        /// Updates the alpha channel with the one from the specified tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        public void UpdateAlphaChannel(ImageTensor tensor)
        {
            var source = tensor.GetChannel(tensor.Channels);
            UpdateChannel(Channels, source);
        }


        /// <summary>
        /// Resizes the ImageTensor
        /// </summary>
        /// <param name="width">The target width in pixels.</param>
        /// <param name="height">The target height in pixels..</param>
        /// <param name="resizeMode">The resize mode.</param>
        public void Resize(int width, int height, ResizeMode resizeMode)
        {
           UpdateTensor(this.ResizeImage(width, height, resizeMode));
        }


        /// <summary>
        /// Clones as ImageTensor.
        /// </summary>
        /// <returns>ImageTensor.</returns>
        public ImageTensor CloneAs()
        {
            return Clone().AsImageTensor();
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

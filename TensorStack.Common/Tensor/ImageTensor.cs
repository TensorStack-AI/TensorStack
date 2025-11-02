// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using TensorStack.Common.Image;

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
            : base(ProcessChannels(tensor), [1, 4, .. tensor.Dimensions[^2..]])
        {
            ThrowIfInvalid();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ImageTensor"/> class.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        public ImageTensor(int height, int width, float fill = 0)
            : base([1, 4, height, width], fill)
        {
            ThrowIfInvalid();
        }


        /// <summary>
        /// Gets the image height.
        /// </summary>
        public int Height => Dimensions[2];

        /// <summary>
        /// Gets the image width.
        /// </summary>
        public int Width => Dimensions[3];


        /// <summary>
        /// Gets a TensorSpan with the specified channels. (1 = Greyscale, 3 = RGB, 4 = RGBA)
        /// </summary>
        /// <param name="count">The channels count.</param>
        /// <returns>TensorSpan&lt;System.Single&gt;.</returns>
        public TensorSpan<float> GetChannels(int channels)
        {
            ArgumentOutOfRangeException.ThrowIfGreaterThan(channels, 4);
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(channels, 0);
            if (channels == 4)
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
            ArgumentOutOfRangeException.ThrowIfGreaterThan(channel, 4);
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
        public void UpdateAlphaChannel(ReadOnlySpan<float> channelData)
        {
            UpdateChannel(4, channelData);
        }


        /// <summary>
        /// Updates the alpha channel with the one from the specified tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        public void UpdateAlphaChannel(ImageTensor tensor)
        {
            UpdateAlphaChannel(tensor.GetChannel(4));
        }


        /// <summary>
        /// Flattens the alpha channel.
        /// </summary>
        public void FlattenAlphaChannel()
        {
            FlattenAlphaChannel(GetChannel(4));
        }


        /// <summary>
        /// Flattens the alpha channel.
        /// </summary>
        /// <param name="alphaChannel">The alpha channel.</param>
        public void FlattenAlphaChannel(ReadOnlySpan<float> alphaChannel)
        {
            var mask = 1f;
            var pixelCount = Height * Width;
            var inputSpan = Memory.Span;
            var outputSpan = Memory.Span;

            var rSpan = inputSpan.Slice(0 * pixelCount, pixelCount);
            var gSpan = inputSpan.Slice(1 * pixelCount, pixelCount);
            var bSpan = inputSpan.Slice(2 * pixelCount, pixelCount);

            var outR = outputSpan.Slice(0 * pixelCount, pixelCount);
            var outG = outputSpan.Slice(1 * pixelCount, pixelCount);
            var outB = outputSpan.Slice(2 * pixelCount, pixelCount);

            for (int i = 0; i < pixelCount; i++)
            {
                float alpha = alphaChannel[i];
                float invAlpha = 1f - alpha;

                outR[i] = rSpan[i] * alpha + mask * invAlpha;
                outG[i] = gSpan[i] * alpha + mask * invAlpha;
                outB[i] = bSpan[i] * alpha + mask * invAlpha;
            }
            OnTensorDataChanged();
        }


        /// <summary>
        /// Resizes the ImageTensor
        /// </summary>
        /// <param name="width">The target width in pixels.</param>
        /// <param name="height">The target height in pixels..</param>
        /// <param name="resizeMode">The resize mode.</param>
        public void Resize(int width, int height, ResizeMode resizeMode, ResizeMethod resizeMethod = ResizeMethod.Bilinear)
        {
            UpdateTensor(this.ResizeImage(width, height, resizeMode, resizeMethod));
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
        /// Gets the RGBA pixel from ImageTensor
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <returns>ImagePixel.</returns>
        /// <exception cref="System.ArgumentOutOfRangeException">Pixel ({x},{y}) out of range ({imageTensor.Width}x{imageTensor.Height}).</exception>
        public ImagePixel GetPixel(int x, int y)
        {
            var span = Memory.Span;
            var stride = Height * Width;
            var pixelIndex = y * Width + x;
            return new ImagePixel(
                span[0 * stride + pixelIndex],
                span[1 * stride + pixelIndex],
                span[2 * stride + pixelIndex],
                span[3 * stride + pixelIndex]
            );
        }


        /// <summary>
        /// Sets the RGBA pixel for ImageTensor
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="color">The color.</param>
        /// <exception cref="System.ArgumentOutOfRangeException">Pixel ({x},{y}) out of range ({imageTensor.Width}x{imageTensor.Height}).</exception>
        public void SetPixel(int x, int y, ImagePixel color)
        {
            var span = Memory.Span;
            var stride = Height * Width;
            var pixelIndex = y * Width + x;
            span[0 * stride + pixelIndex] = color.R;
            span[1 * stride + pixelIndex] = color.G;
            span[2 * stride + pixelIndex] = color.B;
            span[3 * stride + pixelIndex] = color.A;
        }


        /// <summary>
        /// Processes the pixels.
        /// </summary>
        /// <param name="processor">The processor.</param>
        public void ProcessPixels(Func<ImagePixel, ImagePixel> processor)
        {
            var span = Memory.Span;
            var pixelCount = Height * Width;
            var stride = pixelCount;
            var r = span.Slice(0 * stride, stride);
            var g = span.Slice(1 * stride, stride);
            var b = span.Slice(2 * stride, stride);
            var a = span.Slice(3 * stride, stride);
            for (int i = 0; i < pixelCount; i++)
            {
                var px = new ImagePixel(r[i], g[i], b[i], a[i]);
                px = processor(px);
                r[i] = px.R;
                g[i] = px.G;
                b[i] = px.B;
                a[i] = px.A;
            }
        }


        /// <summary>
        /// Throws if Dimensions are invalid.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        protected void ThrowIfInvalid()
        {
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(Height, 0, nameof(Height));
            ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(Width, 0, nameof(Width));
        }


        /// <summary>
        /// Processes the channels ensuring RGBA format.
        /// </summary>
        /// <param name="inputTensor">The input tensor.</param>
        /// <returns>Memory&lt;System.Single&gt;.</returns>
        /// <exception cref="System.ArgumentException">Unsupported channel count</exception>
        private static Memory<float> ProcessChannels(Tensor<float> inputTensor)
        {
            var channels = inputTensor.Dimensions[1];
            if (channels == 4)
                return inputTensor.Memory;

            var height = inputTensor.Dimensions[2];
            var width = inputTensor.Dimensions[3];
            var pixelCount = height * width;
            var output = new float[pixelCount * 4];
            var inputSpan = inputTensor.Memory.Span;
            var outputSpan = output.AsSpan();
            if (channels == 3)
            {
                // Copy RGB channels
                inputSpan.CopyTo(outputSpan[..inputSpan.Length]);

                // Add alpha channel
                outputSpan.Slice(3 * pixelCount, pixelCount).Fill(1f);
                return output;
            }

            if (channels == 1)
            {
                // Copy RGB channels
                var r = outputSpan.Slice(0 * pixelCount, pixelCount);
                var g = outputSpan.Slice(1 * pixelCount, pixelCount);
                var b = outputSpan.Slice(2 * pixelCount, pixelCount);
                inputSpan[..pixelCount].CopyTo(r);
                inputSpan[..pixelCount].CopyTo(g);
                inputSpan[..pixelCount].CopyTo(b);

                // Add alpha channel
                outputSpan.Slice(3 * pixelCount, pixelCount).Fill(1f);
                return output;
            }

            throw new ArgumentException("Unsupported channel count");
        }

    }
}

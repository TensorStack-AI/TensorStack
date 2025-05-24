// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common.Tensor;
using TensorStack.Common.Video;

namespace TensorStack.Video
{
    public static class Extensions
    {
        /// <summary>
        /// Saves the video.
        /// </summary>
        /// <param name="imageFrames">The image frames.</param>
        /// <param name="videoFile">The video file.</param>
        /// <param name="framerate">The framerate.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="videoCodec">The video codec.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public static async Task SaveAync(this IAsyncEnumerable<ImageTensor> imageFrames, string videoFile, float framerate, int width, int height, string videoCodec = "mp4v", CancellationToken cancellationToken = default)
        {
            var frameSize = new Size(width, height);
            var fourcc = VideoWriter.FourCC(videoCodec);
            await VideoService.WriteVideoFramesAsync(imageFrames, videoFile, frameSize, framerate, fourcc, cancellationToken);
        }


        /// <summary>
        /// Saves the video.
        /// </summary>
        /// <param name="imageFrames">The image frames.</param>
        /// <param name="videoFile">The video file.</param>
        /// <param name="framerate">The framerate.</param>
        /// <param name="videoCodec">The video codec.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public static async Task SaveAync(this IAsyncEnumerable<VideoFrame> videoFrames, string videoFile, float framerate, int width, int height, string videoCodec = "mp4v", CancellationToken cancellationToken = default)
        {
            var frameSize = new Size(width, height);
            var fourcc = VideoWriter.FourCC(videoCodec);
            var imageFrames = videoFrames.AsImageTensors(cancellationToken);
            await VideoService.WriteVideoFramesAsync(imageFrames, videoFile, frameSize, framerate, fourcc, cancellationToken);
        }


        /// <summary>
        /// Converts Mat to Tensor.
        /// </summary>
        /// <param name="mat">The mat.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        internal static ImageTensor ToTensor(this Mat mat, Size cropSize = default)
        {
            var cropX = 0;
            var cropY = 0;
            var height = mat.Rows;
            var width = mat.Cols;
            if (width == cropSize.Width)
            {
                cropY = (height - cropSize.Height) / 2;
                height = cropSize.Height;
            }
            else if (height == cropSize.Height)
            {
                cropX = (width - cropSize.Width) / 2;
                width = cropSize.Width;
            }

            var tensor = new ImageTensor([1, 4, height, width]);
            unsafe
            {
                byte* dataPtr = mat.DataPointer;
                for (int y = 0; y < tensor.Height; y++)
                {
                    for (int x = 0; x < tensor.Width; x++)
                    {
                        int pixelIndex = ((y + cropY) * mat.Cols + (x + cropX)) * 3;
                        tensor[0, 0, y, x] = GetFloatValue(dataPtr[pixelIndex + 2]); // R
                        tensor[0, 1, y, x] = GetFloatValue(dataPtr[pixelIndex + 1]); // G
                        tensor[0, 2, y, x] = GetFloatValue(dataPtr[pixelIndex + 0]); // B
                        tensor[0, 3, y, x] = GetFloatValue(byte.MaxValue);           // A
                    }
                }
            }
            return tensor;
        }


        /// <summary>
        /// Converts Tensor to Mat.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>Mat.</returns>
        internal static Mat ToMat(this Tensor<float> tensor)
        {
            var channels = tensor.Dimensions[1];
            var height = tensor.Dimensions[2];
            var width = tensor.Dimensions[3];
            var mat = new Mat(height, width, MatType.CV_8UC3);
            unsafe
            {
                byte* dataPtr = mat.DataPointer;
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int pixelIndex = (y * width + x) * 3;
                        if (channels == 1)
                        {
                            var grayscale = GetByteValue(tensor[0, 0, y, x]);
                            dataPtr[pixelIndex + 2] = grayscale; // R
                            dataPtr[pixelIndex + 1] = grayscale; // G
                            dataPtr[pixelIndex + 0] = grayscale; // B
                        }
                        else
                        {
                            dataPtr[pixelIndex + 2] = GetByteValue(tensor[0, 0, y, x]); // R
                            dataPtr[pixelIndex + 1] = GetByteValue(tensor[0, 1, y, x]); // G
                            dataPtr[pixelIndex + 0] = GetByteValue(tensor[0, 2, y, x]); // B
                        }
                    }
                }
            }
            return mat;
        }


        /// <summary>
        /// Gets the normalized byte value.
        /// </summary>
        /// <param name="value">The value.</param>
        internal static byte GetByteValue(this float value)
        {
            return (byte)Math.Clamp((value + 1.0f) * 0.5f * 255f, 0, 255);
        }


        /// <summary>
        /// Gets the normalized float value.
        /// </summary>
        /// <param name="value">The value.</param>
        internal static float GetFloatValue(this byte value)
        {
            return (value / 255f) * 2.0f - 1.0f;
        }


        /// <summary>
        /// Null if zero.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>System.Nullable&lt;System.Int32&gt;.</returns>
        internal static int? NullIfZero(this int? value)
        {
            if (value.HasValue && value.Value == 0)
                return null;

            return value;
        }


        /// <summary>
        /// Gets the image tensors.
        /// </summary>
        /// <param name="videoFrames">The video frames.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        internal static async IAsyncEnumerable<ImageTensor> AsImageTensors(this IAsyncEnumerable<VideoFrame> videoFrames, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await foreach (var videoFrame in videoFrames.WithCancellation(cancellationToken))
            {
                yield return videoFrame.Frame;
            }
        }
    }
}

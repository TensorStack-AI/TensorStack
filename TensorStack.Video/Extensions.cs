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
        public static async Task SaveAync(this IAsyncEnumerable<ImageTensor> imageFrames, string videoFile, float framerate, int? widthOverride = null, int? heightOverride = null, string videoCodec = "mp4v", CancellationToken cancellationToken = default)
        {
            var videoFrames = imageFrames.AsVideoFrames(framerate, cancellationToken);
            await VideoService.WriteVideoStreamAsync(videoFile, videoFrames, widthOverride, heightOverride, framerate, videoCodec, cancellationToken);
        }


        /// <summary>
        /// Saves the video.
        /// </summary>
        /// <param name="imageFrames">The image frames.</param>
        /// <param name="videoFile">The video file.</param>
        /// <param name="framerate">The framerate.</param>
        /// <param name="videoCodec">The video codec.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public static async Task SaveAync(this IAsyncEnumerable<VideoFrame> videoFrames, string videoFile, int? widthOverride = null, int? heightOverride = null, float? frameRateOverride = null, string videoCodec = "mp4v", CancellationToken cancellationToken = default)
        {
            await VideoService.WriteVideoStreamAsync(videoFile, videoFrames, widthOverride, heightOverride, frameRateOverride, videoCodec, cancellationToken);
        }


        /// <summary>
        /// Saves the video frames processing each frame [Read -> Process -> Write].
        /// Reads an writes are buffered allowing higher processing thoughput
        /// </summary>
        /// <param name="videoInput">The VideoInputStream.</param>
        /// <param name="videoFile">The video file.</param>
        /// <param name="frameProcessor">The frame processor.</param>
        /// <param name="readBuffer">The read buffer (frames).</param>
        /// <param name="writeBuffer">The write buffer (frames).</param>
        /// <param name="widthOverride">The output width override.</param>
        /// <param name="heightOverride">The output height override.</param>
        /// <param name="frameRateOverride">The output frame rate override.</param>
        /// <param name="videoCodec">The video codec.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public static async Task<VideoInputStream> SaveAync(this VideoInputStream videoInput, string videoFile, Func<VideoFrame, Task<VideoFrame>> frameProcessor, int readBuffer = 16, int writeBuffer = 16, int? widthOverride = null, int? heightOverride = null, float? frameRateOverride = null, string videoCodec = "mp4v", CancellationToken cancellationToken = default)
        {
            var videoFrames = videoInput.GetAsync(cancellationToken: cancellationToken);
            await VideoService.WriteVideoStreamAsync(videoFile, videoFrames, frameProcessor, readBuffer, writeBuffer, widthOverride, heightOverride, frameRateOverride, videoCodec, cancellationToken);
            return new VideoInputStream(videoFile, videoCodec);
        }


        /// <summary>
        /// Converts Mat to Tensor.
        /// </summary>
        /// <param name="mat">The mat.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        internal static unsafe ImageTensor ToTensor(this Mat mat, Size cropSize = default)
        {
            int cropX = 0;
            int cropY = 0;
            int height = mat.Rows;
            int width = mat.Cols;

            if (cropSize != default)
            {
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
            }

            var imageTensor = new ImageTensor([1, 4, height, width]);
            var destination = imageTensor.Memory.Span;

            unsafe
            {
                var source = mat.DataPointer;
                int srcStride = mat.Cols * 3;
                int dstStride = height * width;
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int srcIndex = ((y + cropY) * mat.Cols + (x + cropX)) * 3;
                        int dstIndex = y * width + x;

                        destination[0 * dstStride + dstIndex] = GetFloatValue(source[srcIndex + 2]); // R
                        destination[1 * dstStride + dstIndex] = GetFloatValue(source[srcIndex + 1]); // G
                        destination[2 * dstStride + dstIndex] = GetFloatValue(source[srcIndex + 0]); // B
                        destination[3 * dstStride + dstIndex] = GetFloatValue(byte.MaxValue);        // A
                    }
                }
            }

            return imageTensor;
        }


        /// <summary>
        /// Converts Tensor to OpenCv Matrix.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>Mat.</returns>
        internal static unsafe Mat ToMatrix(this Tensor<float> tensor)
        {
            var channels = tensor.Dimensions[1];
            var height = tensor.Dimensions[2];
            var width = tensor.Dimensions[3];

            var matrix = new Mat(height, width, MatType.CV_8UC3);
            var source = tensor.Span;
            var destination = matrix.DataPointer;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int offset = y * width + x;

                    if (channels == 1)
                    {
                        byte gray = GetByteValue(source[offset]);
                        destination[offset * 3 + 0] = gray; // B
                        destination[offset * 3 + 1] = gray; // G
                        destination[offset * 3 + 2] = gray; // R
                    }
                    else
                    {
                        destination[offset * 3 + 0] = GetByteValue(source[2 * width * height + offset]); // B
                        destination[offset * 3 + 1] = GetByteValue(source[1 * width * height + offset]); // G
                        destination[offset * 3 + 2] = GetByteValue(source[0 * width * height + offset]); // R
                    }
                }
            }

            return matrix;
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


        internal static async IAsyncEnumerable<VideoFrame> AsVideoFrames(this IAsyncEnumerable<ImageTensor> videoFrames, float frameRate, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var frameIndex = 0;
            await foreach (var videoFrame in videoFrames.WithCancellation(cancellationToken))
            {
                yield return new VideoFrame(frameIndex++, videoFrame, frameRate);
            }
        }
    }
}

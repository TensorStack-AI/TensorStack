// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.Common.Video;

namespace TensorStack.Video
{
    public static class Extensions
    {
        /// <summary>
        /// Saves the video.
        /// </summary>
        /// <param name="videoTensor">The video tensor.</param>
        /// <param name="videoFile">The video file.</param>
        /// <param name="framerate">The framerate.</param>
        /// <param name="videoCodec">The video codec.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public static async Task SaveAync(this VideoTensor videoTensor, string videoFile, float? framerate = default, string videoCodec = "mp4v", CancellationToken cancellationToken = default)
        {
            var frames = videoTensor
                .Split()
                .Select(frame => new ImageTensor(frame))
                .ToAsyncEnumerable();
            await SaveAync(frames, videoFile, framerate ?? videoTensor.FrameRate, videoCodec, cancellationToken);
        }


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
        public static async Task SaveAync(this IAsyncEnumerable<ImageTensor> imageFrames, string videoFile, float framerate, string videoCodec = "mp4v", CancellationToken cancellationToken = default)
        {
            var fourcc = VideoWriter.FourCC(videoCodec);
            var firstFrame = await imageFrames.FirstAsync(cancellationToken);
            var fullSequence = CacheFirstAndIterate(firstFrame, imageFrames, cancellationToken);
            var frameSize = new Size(firstFrame.Width, firstFrame.Height);
            await WriteVideoFramesAsync(fullSequence, videoFile, frameSize, framerate, fourcc, cancellationToken);
        }


        /// <summary>
        /// Saves the video.
        /// </summary>
        /// <param name="imageFrames">The image frames.</param>
        /// <param name="videoFile">The video file.</param>
        /// <param name="framerate">The framerate.</param>
        /// <param name="videoCodec">The video codec.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public static async Task SaveAync(this IAsyncEnumerable<VideoFrame> imageFrames, string videoFile, float? framerate = default, string videoCodec = "mp4v", CancellationToken cancellationToken = default)
        {
            var fourcc = VideoWriter.FourCC(videoCodec);
            var firstFrame = await imageFrames.FirstAsync(cancellationToken);
            var fullSequence = CacheFirstAndIterate(firstFrame, imageFrames, cancellationToken);
            var frameSize = new Size(firstFrame.Width, firstFrame.Height);
            var outframeRate = framerate ?? firstFrame.SourceFrameRate;
            await WriteVideoFramesAsync(fullSequence, videoFile, frameSize, outframeRate, fourcc, cancellationToken);
        }


        /// <summary>
        /// Get a VideoTensor from the VideoInput
        /// </summary>
        /// <param name="videoStream">The video stream.</param>
        /// <param name="framerate">The framerate.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;VideoTensor&gt; representing the asynchronous operation.</returns>
        internal static async Task<VideoTensor> GetVideoTensorAsync(this VideoInput videoStream, float? framerate = default, int? width = default, int? height = default, CancellationToken cancellationToken = default)
        {
            var buffered = await videoStream
                .GetStreamAsync(width: width, height: height, cancellationToken: cancellationToken)
                .Select(x => x.Frame)
                .ToArrayAsync(cancellationToken);
            return new VideoTensor(buffered.Join(), videoStream.FrameRate);
        }


        /// <summary>
        /// Save video stream as an asynchronous operation.
        /// </summary>
        /// <param name="videoFile">The video file.</param>
        /// <param name="imageFrames">The image frames.</param>
        /// <param name="videoCodec">The video codec.</param>
        /// <param name="framerate">The framerate.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        internal static async Task WriteVideoStreamAsync(IAsyncEnumerable<VideoFrame> imageFrames, string videoFile, string videoCodec, float? framerate = default, int? width = default, int? height = default, CancellationToken cancellationToken = default)
        {
            var fourcc = VideoWriter.FourCC(videoCodec);
            var firstFrame = await imageFrames.FirstAsync(cancellationToken);
            var fullSequence = CacheFirstAndIterate(firstFrame, imageFrames, cancellationToken);
            var outputHeight = height.NullIfZero() ?? firstFrame.Frame.Dimensions[2];
            var outputWidth = width.NullIfZero() ?? firstFrame.Frame.Dimensions[3];
            var outputFramerate = framerate ?? firstFrame.SourceFrameRate;
            var frameSize = new Size(outputWidth, outputHeight);
            await WriteVideoFramesAsync(fullSequence, videoFile, frameSize, outputFramerate, fourcc, cancellationToken);
        }


        /// <summary>
        /// Get video stream as an asynchronous operation.
        /// </summary>
        /// <param name="videoFile">The video file.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;IAsyncEnumerable`1&gt; representing the asynchronous operation.</returns>
        /// <exception cref="System.Exception">Failed to open video file.</exception>
        internal static async IAsyncEnumerable<VideoFrame> ReadVideoFramesAsync(string videoFile, float? frameRate = default, int? width = default, int? height = default, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            using (var videoReader = new VideoCapture(videoFile))
            {
                if (!videoReader.IsOpened())
                    throw new Exception("Failed to open video file.");

                var frameCount = 0;
                var emptySize = new Size(0, 0);
                var outframeRate = (float)(frameRate.HasValue ? Math.Min(frameRate.Value, videoReader.Fps) : videoReader.Fps);
                var frameSkipInterval = frameRate.HasValue ? (int)(Math.Round(videoReader.Fps) / Math.Min(Math.Round(frameRate.Value), Math.Round(videoReader.Fps))) : 1;
                var isScaleRequired = width.NullIfZero().HasValue || height.NullIfZero().HasValue;
                var scaleX = (width.NullIfZero() ?? videoReader.FrameWidth) / (double)videoReader.FrameWidth;
                var scaleY = (height.NullIfZero() ?? videoReader.FrameHeight) / (double)videoReader.FrameHeight;
                var scaleFactor = scaleX < 1 && scaleY < 1 ? Math.Max(scaleX, scaleY) : Math.Min(scaleX, scaleY);
                using (var frame = new Mat())
                {
                    while (true)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        videoReader.Read(frame);
                        if (frame.Empty())
                            break;

                        if (frameCount % frameSkipInterval == 0)
                        {
                            if (isScaleRequired)
                                Cv2.Resize(frame, frame, emptySize, scaleFactor, scaleFactor);

                            yield return new VideoFrame(frame.ToTensor(), outframeRate);
                        }

                        frameCount++;
                    }
                }
                await Task.Yield();
            }
        }


        /// <summary>
        /// Write video frames to disk.
        /// </summary>
        /// <param name="videoFile">The video file.</param>
        /// <param name="framerate">The framerate.</param>
        /// <param name="fourcc">The fourcc.</param>
        /// <param name="videoFrames">The full sequence.</param>
        /// <param name="frameSize">Size of the frame.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        private static async Task WriteVideoFramesAsync(IAsyncEnumerable<ImageTensor> videoFrames, string videoFile, Size frameSize, float framerate, int fourcc, CancellationToken cancellationToken)
        {
            await Task.Run(async () =>
            {
                using (var writer = new VideoWriter(videoFile, fourcc, framerate, frameSize))
                {
                    if (!writer.IsOpened())
                        throw new Exception("Failed to open VideoWriter..");

                    await foreach (var imageFrame in videoFrames)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        using (var mat = imageFrame.ToMat())
                        {
                            writer.Write(mat);
                        }
                    }
                }
            }, cancellationToken);
        }


        /// <summary>
        /// Caches the first item and iterate.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="firstItem">The first item.</param>
        /// <param name="remainingItems">The remaining items.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>IAsyncEnumerable&lt;T&gt;.</returns>
        private static async IAsyncEnumerable<ImageTensor> CacheFirstAndIterate(VideoFrame firstItem, IAsyncEnumerable<VideoFrame> remainingItems, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            yield return firstItem.Frame;
            await foreach (var item in remainingItems.WithCancellation(cancellationToken))
            {
                yield return item.Frame;
            }
        }


        /// <summary>
        /// Caches the first item and iterate.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="firstItem">The first item.</param>
        /// <param name="remainingItems">The remaining items.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>IAsyncEnumerable&lt;T&gt;.</returns>
        private static async IAsyncEnumerable<ImageTensor> CacheFirstAndIterate(ImageTensor firstItem, IAsyncEnumerable<ImageTensor> remainingItems, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            yield return firstItem;
            await foreach (var item in remainingItems.WithCancellation(cancellationToken))
            {
                yield return item;
            }
        }


        /// <summary>
        /// Converts Mat to Tensor.
        /// </summary>
        /// <param name="mat">The mat.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        internal static Tensor<float> ToTensor(this Mat mat)
        {
            var height = mat.Rows;
            var width = mat.Cols;
            var tensor = new Tensor<float>(new[] { 1, 3, height, width });
            unsafe
            {
                byte* dataPtr = mat.DataPointer;
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int pixelIndex = (y * width + x) * 3;
                        tensor[0, 0, y, x] = GetFloatValue(dataPtr[pixelIndex + 2]); // R
                        tensor[0, 1, y, x] = GetFloatValue(dataPtr[pixelIndex + 1]); // G
                        tensor[0, 2, y, x] = GetFloatValue(dataPtr[pixelIndex + 0]); // B
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
        private static byte GetByteValue(float value)
        {
            return (byte)Math.Clamp((value + 1.0f) * 0.5f * 255f, 0, 255);
        }


        /// <summary>
        /// Gets the normalized float value.
        /// </summary>
        /// <param name="value">The value.</param>
        private static float GetFloatValue(byte value)
        {
            return (value / 255f) * 2.0f - 1.0f;
        }


        private static int? NullIfZero(this int? value)
        {
            if (value.HasValue && value.Value == 0)
                return null;

            return value;
        }
    }
}

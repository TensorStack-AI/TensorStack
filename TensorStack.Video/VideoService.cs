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
    public static class VideoService
    {

        /// <summary>
        /// Load the video information.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <returns>VideoInfo.</returns>
        /// <exception cref="System.Exception">Failed to open video file.</exception>
        public static VideoInfo LoadVideoInfo(string filename)
        {
            using (var videoReader = new VideoCapture(filename))
            {
                if (!videoReader.IsOpened())
                    throw new Exception("Failed to open video file.");

                return new VideoInfo(filename, videoReader.FrameWidth, videoReader.FrameHeight, (float)videoReader.Fps, videoReader.FrameCount);
            }
        }


        /// <summary>
        /// Loads the VideoTensor from file.
        /// </summary>
        /// <param name="videoFile">The video file.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <returns>VideoTensor.</returns>
        public static VideoTensor LoadVideoTensor(string videoFile, int? width = default, int? height = default, float? frameRate = default, ResizeMode resizeMode = ResizeMode.Crop)
        {
            return ReadVideo(videoFile, width, height, frameRate, resizeMode);
        }


        /// <summary>
        /// Loads the VideoTensor from file asynchronous.
        /// </summary>
        /// <param name="videoFile">The video file.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>Task&lt;VideoTensor&gt;.</returns>
        public static Task<VideoTensor> LoadVideoTensorAsync(string videoFile, int? width = default, int? height = default, float? frameRate = default, ResizeMode resizeMode = ResizeMode.Crop, CancellationToken cancellationToken = default)
        {
            return Task.Run(() => ReadVideo(videoFile, width, height, frameRate, resizeMode, cancellationToken));
        }


        /// <summary>
        /// Saves the video.
        /// </summary>
        /// <param name="videoTensor">The video tensor.</param>
        /// <param name="videoFile">The video file.</param>
        /// <param name="framerate">The framerate.</param>
        /// <param name="videoCodec">The video codec.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public static async Task SaveVideoTensorAync(VideoTensor videoTensor, string videoFile, float? framerate = default, string videoCodec = "mp4v", CancellationToken cancellationToken = default)
        {
            var frames = videoTensor
                .Split()
                .Select(frame => new ImageTensor(frame))
                .ToAsyncEnumerable();
            await frames.SaveAync(videoFile, framerate ?? videoTensor.FrameRate, videoTensor.Width, videoTensor.Height, videoCodec, cancellationToken);
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
        internal static async Task WriteVideoStreamAsync(IAsyncEnumerable<VideoFrame> videoFrames, string videoFile, float framerate, int width, int height, string videoCodec = "mp4v", CancellationToken cancellationToken = default)
        {
            var frameSize = new Size(width, height);
            var fourcc = VideoWriter.FourCC(videoCodec);
            var imageFrames = videoFrames.AsImageTensors(cancellationToken);
            await WriteVideoFramesAsync(imageFrames, videoFile, frameSize, framerate, fourcc, cancellationToken);
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
        internal static async IAsyncEnumerable<VideoFrame> ReadStreamAsync(string videoFile, float? frameRate = default, int? width = default, int? height = default, ResizeMode resizeMode = ResizeMode.Stretch, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            using (var videoReader = new VideoCapture(videoFile))
            {
                if (!videoReader.IsOpened())
                    throw new Exception("Failed to open video file.");

                await Task.Yield();
                var frameCount = 0;
                var videoSize = new Size(videoReader.FrameWidth, videoReader.FrameHeight);
                var videoNewSize = GetNewVideoSize(width, height, videoSize, resizeMode);
                var videoCropSize = GetCropVideoSize(width, height, videoNewSize, resizeMode);
                var videoframeRate = GetVideoFrameRate(videoReader.Fps, frameRate);
                var frameSkipInterval = GetFrameInterval(videoReader.Fps, frameRate);
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
                            if (videoSize != videoNewSize)
                                Cv2.Resize(frame, frame, videoNewSize);

                            yield return new VideoFrame(frame.ToTensor(videoCropSize), videoframeRate);
                        }
                        frameCount++;
                    }
                }
            }
        }


        /// <summary>
        /// Creates a new VideoTensor (in-memory)
        /// </summary>
        /// <param name="videoFile">The video file.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <param name="resizeMode">The resize mode.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>VideoTensor.</returns>
        /// <exception cref="System.Exception">Failed to open video file.</exception>
        internal static VideoTensor ReadVideo(string videoFile, int? width = default, int? height = default, float? frameRate = default, ResizeMode resizeMode = ResizeMode.Stretch, CancellationToken cancellationToken = default)
        {
            using (var videoReader = new VideoCapture(videoFile))
            {
                if (!videoReader.IsOpened())
                    throw new Exception("Failed to open video file.");

                var frameCount = 0;
                var result = new List<ImageTensor>();
                var videoSize = new Size(videoReader.FrameWidth, videoReader.FrameHeight);
                var videoNewSize = GetNewVideoSize(width, height, videoSize, resizeMode);
                var videoCropSize = GetCropVideoSize(width, height, videoNewSize, resizeMode);
                var videoframeRate = GetVideoFrameRate(videoReader.Fps, frameRate);
                var frameSkipInterval = GetFrameInterval(videoReader.Fps, frameRate);
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
                            if (videoSize != videoNewSize)
                                Cv2.Resize(frame, frame, videoNewSize);

                            result.Add(frame.ToTensor(videoCropSize));
                        }
                        frameCount++;
                    }
                }
                return new VideoTensor(result.Join(), videoframeRate);
            }
        }


        /// <summary>
        /// Gets the video frame rate.
        /// </summary>
        /// <param name="framerate">The framerate.</param>
        /// <param name="newFramerate">The new framerate.</param>
        /// <returns>System.Single.</returns>
        private static float GetVideoFrameRate(double framerate, float? newFramerate)
        {
            return (float)(newFramerate.HasValue ? Math.Min(newFramerate.Value, framerate) : framerate);
        }


        /// <summary>
        /// Gets the video size scaled to the aspect and ResizeMode
        /// </summary>
        /// <param name="cropWidth">Width of the crop.</param>
        /// <param name="cropHeight">Height of the crop.</param>
        /// <param name="currentSize">Size of the current.</param>
        /// <param name="resizeMode">The resize mode.</param>
        /// <returns>Size.</returns>
        private static Size GetNewVideoSize(int? cropWidth, int? cropHeight, Size currentSize, ResizeMode resizeMode)
        {
            var width = cropWidth.NullIfZero();
            var height = cropHeight.NullIfZero();
            if (!width.HasValue && !height.HasValue)
                return currentSize;

            if (resizeMode == ResizeMode.Stretch)
            {
                if (width.HasValue && height.HasValue)
                    return new Size(width.Value, height.Value);
                if (width.HasValue)
                    return new Size(width.Value, currentSize.Height);
                if (height.HasValue)
                    return new Size(currentSize.Width, height.Value);
            }

            if (width.HasValue && height.HasValue)
            {
                var scaleX = (float)width.Value / currentSize.Width;
                var scaleY = (float)height.Value / currentSize.Height;
                var scale = Math.Max(scaleX, scaleY);
                return new Size((int)(currentSize.Width * scale), (int)(currentSize.Height * scale));
            }
            else if (width.HasValue)
            {
                var scaleX = (float)width.Value / currentSize.Width;
                return new Size((int)(currentSize.Width * scaleX), (int)(currentSize.Height * scaleX));
            }
            else if (height.HasValue)
            {
                var scaleY = (float)height.Value / currentSize.Height;
                return new Size((int)(currentSize.Width * scaleY), (int)(currentSize.Height * scaleY));
            }
            return currentSize;
        }


        /// <summary>
        /// Gets the size of the crop video.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="currentSize">Size of the current.</param>
        /// <param name="resizeMode">The resize mode.</param>
        /// <returns>Size.</returns>
        private static Size GetCropVideoSize(int? cropWidth, int? cropHeight, Size currentSize, ResizeMode resizeMode)
        {
            var cropSize = default(Size);
            if (resizeMode == ResizeMode.Crop)
            {
                var width = cropWidth.NullIfZero();
                var height = cropHeight.NullIfZero();
                if (width.HasValue || height.HasValue)
                {
                    if (width.HasValue && height.HasValue)
                        cropSize = new Size(width.Value, height.Value);
                    else if (width.HasValue)
                        cropSize = new Size(width.Value, currentSize.Height);
                    else if (height.HasValue)
                        cropSize = new Size(currentSize.Width, height.Value);
                }
            }
            return cropSize;
        }


        /// <summary>
        /// Gets the frame interval.
        /// </summary>
        /// <param name="framerate">The framerate.</param>
        /// <param name="newFramerate">The new framerate.</param>
        /// <returns>System.Int32.</returns>
        private static int GetFrameInterval(double framerate, float? newFramerate)
        {
            if (!newFramerate.HasValue)
                return 1;

            return (int)(Math.Round(framerate) / Math.Min(Math.Round(newFramerate.Value), Math.Round(framerate)));
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
        internal static async Task WriteVideoFramesAsync(IAsyncEnumerable<ImageTensor> videoFrames, string videoFile, Size frameSize, float framerate, int fourcc, CancellationToken cancellationToken)
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
    }
}

// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.Common.Video;

namespace TensorStack.Video
{
    public static class VideoManager
    {
        /// <summary>
        /// Load the video information.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="thumbSize">Size of the thumb (largest side).</param>
        /// <param name="thumbPos">The seek position % to capture the thumbnail.</param>
        /// <returns>VideoInfo.</returns>
        /// <exception cref="System.Exception">Failed to open video file.</exception>
        /// <exception cref="System.Exception">Failed to read video frame.</exception>
        public static VideoInfo LoadVideoInfo(string filename, int thumbSize = 100, float thumbPos = 0.10f)
        {
            using (var videoReader = new VideoCapture(filename))
            {
                if (!videoReader.IsOpened())
                    throw new Exception("Failed to open video file.");

                // Seek to n% for thumbnail
                var totalFrames = (int)videoReader.Get(VideoCaptureProperties.FrameCount);
                var targetFrame = (int)Math.Clamp(totalFrames * thumbPos, 0, totalFrames - 1);
                videoReader.Set(VideoCaptureProperties.PosFrames, targetFrame);

                using (var thumbFrame = new Mat())
                {
                    if (!videoReader.Read(thumbFrame) || thumbFrame.Empty())
                        throw new Exception("Failed to read video frame.");

                    Cv2.Resize(thumbFrame, thumbFrame, GetNewVideoSize(thumbSize, default, thumbFrame.Size(), ResizeMode.Crop));
                    return new VideoInfo
                    {
                        FileName = filename,
                        Width = videoReader.FrameWidth,
                        Height = videoReader.FrameHeight,
                        FrameRate = (float)videoReader.Fps,
                        FrameCount = videoReader.FrameCount,
                        Thumbnail = thumbFrame.ToTensor(),
                        VideoCodec = videoReader.FourCC
                    };
                }
            }
        }


        /// <summary>
        /// Load the video information.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="thumbSize">Size of the thumb (largest side).</param>
        /// <param name="thumbPos">The seek position % to capture the thumbnail.</param>
        /// <returns>VideoInfo.</returns>
        /// <exception cref="System.Exception">Failed to open video file.</exception>
        /// <exception cref="System.Exception">Failed to read video frame.</exception>
        public static Task<VideoInfo> LoadVideoInfoAsync(string filename, int thumbSize = 100, float thumbPos = 0.10f)
        {
            return Task.Run(() => LoadVideoInfo(filename, thumbSize, thumbPos));
        }


        /// <summary>
        /// Loads the VideoTensor from file.
        /// </summary>
        /// <param name="videoFile">The video file.</param>
        /// <param name="widthOverride">The width.</param>
        /// <param name="heightOverride">The height.</param>
        ///  <param name="frameRateOverride">The frame rate.</param>
        /// <returns>VideoTensor.</returns>
        public static VideoTensor LoadVideoTensor(string videoFile, int? widthOverride = default, int? heightOverride = default, float? frameRateOverride = default, ResizeMode resizeMode = ResizeMode.Crop)
        {
            return ReadVideo(videoFile, widthOverride, heightOverride, frameRateOverride, resizeMode);
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
        public static Task<VideoTensor> LoadVideoTensorAsync(string videoFile, int? widthOverride = default, int? heightOverride = default, float? frameRateOverride = default, ResizeMode resizeMode = ResizeMode.Crop, CancellationToken cancellationToken = default)
        {
            return Task.Run(() => ReadVideo(videoFile, widthOverride, heightOverride, frameRateOverride, resizeMode, cancellationToken));
        }


        /// <summary>
        /// Saves the video.
        /// </summary>
        /// <param name="videoTensor">The video tensor.</param>
        /// <param name="videoFile">The video file.</param>
        /// <param name="framerate">The framerate.</param>
        /// <param name="videoCodec">The video codec.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        internal static async Task SaveVideoTensorAync(string videoFile, VideoTensor videoTensor, string videoCodec = "mp4v", float? frameRateOverride = default, CancellationToken cancellationToken = default)
        {
            var frameRate = frameRateOverride ?? videoTensor.FrameRate;
            var videoFrames = videoTensor
                .Split()
                .Select((frame, i) => new VideoFrame(i, frame, frameRate))
                .ToAsyncEnumerable();

            await WriteVideoStreamAsync(videoFile, videoFrames, videoCodec, cancellationToken: cancellationToken);
        }


        /// <summary>
        /// Save video stream as an asynchronous operation.
        /// </summary>
        /// <param name="videoFile">The video file.</param>
        /// <param name="videoFrames">The image frames.</param>
        /// <param name="videoCodec">The video codec.</param>
        /// <param name="widthOverride">The width.</param>
        /// <param name="heightOverride">The height.</param>
        /// <param name="framerateOverride">The framerate.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        public static async Task WriteVideoStreamAsync(string videoFile, IAsyncEnumerable<VideoFrame> videoFrames, string videoCodec = "mp4v", int? widthOverride = null, int? heightOverride = null, float? frameRateOverride = null, CancellationToken cancellationToken = default)
        {
            var fourcc = VideoWriter.FourCC(videoCodec);
            await WriteVideoFramesAsync(videoFile, videoFrames, fourcc, widthOverride, heightOverride, frameRateOverride, cancellationToken);
        }


        /// <summary>
        /// Write video stream with buffered read/write
        /// </summary>
        /// <param name="videoFile">The video file.</param>
        /// <param name="videoFrames">The video frames.</param>
        /// <param name="frameProcessor">The frame processor.</param>
        /// <param name="readBuffer">The read buffer.</param>
        /// <param name="writeBuffer">The write buffer.</param>
        /// <param name="widthOverride">The width override.</param>
        /// <param name="heightOverride">The height override.</param>
        /// <param name="frameRateOverride">The frame rate override.</param>
        /// <param name="videoCodec">The video codec.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public static async Task WriteVideoStreamAsync(string videoFile, IAsyncEnumerable<VideoFrame> videoFrames, Func<VideoFrame, Task<VideoFrame>> frameProcessor, int readBuffer = 16, int writeBuffer = 16, string videoCodec = "mp4v", int? widthOverride = null, int? heightOverride = null, float? frameRateOverride = null, CancellationToken cancellationToken = default)
        {
            var fourcc = VideoWriter.FourCC(videoCodec);
            await WriteVideoFramesAsync(videoFile, videoFrames, frameProcessor, readBuffer, writeBuffer, fourcc, widthOverride, heightOverride, frameRateOverride, cancellationToken);
        }


        /// <summary>
        /// Get video stream as an asynchronous operation.
        /// </summary>
        /// <param name="videoFile">The video file.</param>
        /// <param name="frameRateOverride">The frame rate.</param>
        /// <param name="widthOverride">The width.</param>
        /// <param name="heightOverride">The height.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;IAsyncEnumerable`1&gt; representing the asynchronous operation.</returns>
        /// <exception cref="System.Exception">Failed to open video file.</exception>
        internal static async IAsyncEnumerable<VideoFrame> ReadStreamAsync(string videoFile, float? frameRateOverride = default, int? widthOverride = default, int? heightOverride = default, ResizeMode resizeMode = ResizeMode.Stretch, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            using (var videoReader = new VideoCapture(videoFile))
            {
                if (!videoReader.IsOpened())
                    throw new Exception("Failed to open video file.");

                await Task.Yield();
                var frameCount = 0;
                var frameIndex = 0;
                var videoSize = new Size(videoReader.FrameWidth, videoReader.FrameHeight);
                var videoNewSize = GetNewVideoSize(widthOverride, heightOverride, videoSize, resizeMode);
                var videoCropSize = GetCropVideoSize(widthOverride, heightOverride, videoNewSize, resizeMode);
                var videoframeRate = GetVideoFrameRate(videoReader.Fps, frameRateOverride);
                var frameSkipInterval = GetFrameInterval(videoReader.Fps, frameRateOverride);
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

                            yield return new VideoFrame(frameIndex++, frame.ToTensor(videoCropSize), videoframeRate);
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
        internal static VideoTensor ReadVideo(string videoFile, int? widthOverride = default, int? heightOverride = default, float? frameRateOverride = default, ResizeMode resizeMode = ResizeMode.Stretch, CancellationToken cancellationToken = default)
        {
            using (var videoReader = new VideoCapture(videoFile))
            {
                if (!videoReader.IsOpened())
                    throw new Exception("Failed to open video file.");

                var frameCount = 0;
                var result = new List<ImageTensor>();
                var videoSize = new Size(videoReader.FrameWidth, videoReader.FrameHeight);
                var videoNewSize = GetNewVideoSize(widthOverride, heightOverride, videoSize, resizeMode);
                var videoCropSize = GetCropVideoSize(widthOverride, heightOverride, videoNewSize, resizeMode);
                var videoframeRate = GetVideoFrameRate(videoReader.Fps, frameRateOverride);
                var frameSkipInterval = GetFrameInterval(videoReader.Fps, frameRateOverride);
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
        /// Writes the VideoFrames to file
        /// </summary>
        /// <param name="videoOutputFile">The video output file.</param>
        /// <param name="videoFrames">The video frames.</param>
        /// <param name="fourcc">The fourcc.</param>
        /// <param name="widthOverride">The width override.</param>
        /// <param name="heightOverride">The height override.</param>
        /// <param name="frameRateOverride">The frame rate override.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        private static async Task WriteVideoFramesAsync(string videoOutputFile, IAsyncEnumerable<VideoFrame> videoFrames, int fourcc, int? widthOverride = null, int? heightOverride = null, float? frameRateOverride = null, CancellationToken cancellationToken = default)
        {
            await Task.Run(async () =>
            {
                await using var enumerator = videoFrames.GetAsyncEnumerator(cancellationToken);

                if (!await enumerator.MoveNextAsync())
                    throw new Exception("No frames to write.");

                var firstFrame = enumerator.Current;
                var frameSize = new Size(widthOverride ?? firstFrame.Width, heightOverride ?? firstFrame.Height);
                var frameRate = frameRateOverride ?? firstFrame.SourceFrameRate;

                using (var writer = new VideoWriter(videoOutputFile, fourcc, frameRate, frameSize))
                {
                    if (!writer.IsOpened())
                        throw new Exception("Failed to open VideoWriter..");

                    // Write first
                    using (var matrix = firstFrame.Frame.ToMatrix())
                        writer.Write(matrix);

                    // Write the rest
                    while (await enumerator.MoveNextAsync())
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        using (var matrix = enumerator.Current.Frame.ToMatrix())
                            writer.Write(matrix);
                    }
                }
            }, cancellationToken);
        }


        /// <summary>
        /// Reads frames, processes frames and write frames asynchronously with buffering
        /// </summary>
        /// <param name="videoOutputFile">The output video file.</param>
        /// <param name="videoFrames">The video frames.</param>
        /// <param name="frameProcessor">The frame processor.</param>
        /// <param name="readBuffer">The read buffer.</param>
        /// <param name="writeBuffer">The write buffer.</param>
        /// <param name="fourcc">The fourcc.</param>
        /// <param name="widthOverride">The output width override.</param>
        /// <param name="heightOverride">The output height override.</param>
        /// <param name="frameRateOverride">The output framerate override.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        private static async Task WriteVideoFramesAsync(string videoOutputFile, IAsyncEnumerable<VideoFrame> videoFrames, Func<VideoFrame, Task<VideoFrame>> frameProcessor, int readBuffer, int writeBuffer, int fourcc, int? widthOverride = default, int? heightOverride = default, float? frameRateOverride = default, CancellationToken cancellationToken = default)
        {
            var readChannel = Channel.CreateBounded<VideoFrame>(readBuffer);
            var writeChannel = Channel.CreateBounded<VideoFrame>(writeBuffer);

            // Read frames
            var readerTask = Task.Run(async () =>
            {
                await foreach (var frame in videoFrames.WithCancellation(cancellationToken))
                {
                    await readChannel.Writer.WriteAsync(frame, cancellationToken);
                }
                readChannel.Writer.Complete();
            }, cancellationToken);


            // Process Frames
            var processTask = Task.Run(async () =>
            {
                await foreach (var frame in readChannel.Reader.ReadAllAsync(cancellationToken))
                {
                    var processedFrame = await frameProcessor(frame);
                    await writeChannel.Writer.WriteAsync(processedFrame, cancellationToken);
                }
                writeChannel.Writer.Complete();
            }, cancellationToken);


            // Write Frames
            var writeFrames = writeChannel.Reader.ReadAllAsync(cancellationToken);
            var writerTask = WriteVideoFramesAsync(videoOutputFile, writeFrames, fourcc, widthOverride, heightOverride, frameRateOverride, cancellationToken);

            // Block
            await Task.WhenAll(readerTask, processTask, writerTask);
        }



        /// <summary>
        /// Converts Matrix to Tensor.
        /// </summary>
        /// <param name="matrix">The matrix.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        internal static unsafe ImageTensor ToTensor(this Mat matrix, Size cropSize = default)
        {
            int cropX = 0;
            int cropY = 0;
            int height = matrix.Rows;
            int width = matrix.Cols;

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
                var source = matrix.DataPointer;
                int srcStride = matrix.Cols * 3;
                int dstStride = height * width;
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int srcIndex = ((y + cropY) * matrix.Cols + (x + cropX)) * 3;
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
    }
}

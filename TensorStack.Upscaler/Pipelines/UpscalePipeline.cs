﻿// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Image;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.Common.Video;
using TensorStack.Upscaler.Common;
using TensorStack.Upscaler.Models;

namespace TensorStack.Upscaler.Pipelines
{
    /// <summary>
    /// Basic UpscalePipeline. This class cannot be inherited.
    /// </summary>
    public sealed class UpscalePipeline :
          IPipeline<ImageTensor, UpscaleImageOptions>,
          IPipeline<VideoTensor, UpscaleVideoOptions>,
          IPipelineStream<VideoFrame, UpscaleStreamOptions>
    {
        private readonly UpscalerModel _upscaleModel;

        /// <summary>
        /// Initializes a new instance of the <see cref="UpscalePipeline"/> class.
        /// </summary>
        /// <param name="upscaleModel">The upscale model.</param>
        public UpscalePipeline(UpscalerModel upscaleModel)
        {
            _upscaleModel = upscaleModel;
        }


        /// <summary>
        /// Loads the pipeline.
        /// </summary>
        /// <returns>A Task representing the asynchronous operation.</returns>
        public async Task LoadAsync(CancellationToken cancellationToken = default)
        {
            await _upscaleModel.LoadAsync();
        }


        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        /// <returns>A Task representing the asynchronous operation.</returns>
        public async Task UnloadAsync(CancellationToken cancellationToken = default)
        {
            await _upscaleModel.UnloadAsync();
        }


        /// <summary>
        /// Run the pipeline ImageTensor to ImageTensor function with the specified UpscaleOptions
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;ImageTensor&gt; representing the asynchronous operation.</returns>
        public async Task<ImageTensor> RunAsync(UpscaleImageOptions options, IProgress<RunProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            var timestamp = RunProgress.GetTimestamp();
            if (_upscaleModel.Normalization == Normalization.ZeroToOne)
                options.Image.NormalizeZeroToOne();

            var resultTensor = await UpscaleInternalAsync(options.Image, options, cancellationToken);
            if (_upscaleModel.Normalization == Normalization.ZeroToOne)
            {
                options.Image.NormalizeOneToOne();
                resultTensor.NormalizeOneToOne();
            }
            progressCallback?.Report(new RunProgress(timestamp));
            return resultTensor;
        }


        /// <summary>
        /// Run the pipeline VideoTensor to VideoTensor function with the specified UpscaleOptions
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="inputVideo">The input video.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;VideoTensor&gt; representing the asynchronous operation.</returns>
        public async Task<VideoTensor> RunAsync(UpscaleVideoOptions options, IProgress<RunProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            var timestamp = RunProgress.GetTimestamp();
            if (_upscaleModel.Normalization == Normalization.ZeroToOne)
                options.Video.NormalizeZeroToOne();

            var results = new List<ImageTensor>();
            foreach (var frame in options.Video.GetFrames())
            {
                var frameTime = Stopwatch.GetTimestamp();
                var resultTensor = await UpscaleInternalAsync(frame, options, cancellationToken);
                results.Add(resultTensor);
                progressCallback?.Report(new RunProgress(results.Count, options.Video.Frames, frameTime));
            }

            var resultVideoTensor = new VideoTensor(results.Join(), options.Video.FrameRate);
            if (_upscaleModel.Normalization == Normalization.ZeroToOne)
            {
                options.Video.NormalizeOneToOne();
                resultVideoTensor.NormalizeOneToOne();
            }
            progressCallback?.Report(new RunProgress(timestamp));
            return resultVideoTensor;
        }


        /// <summary>
        /// Run the pipeline VideoFrame stream function
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="inputVideoStream">The input video stream.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;IAsyncEnumerable`1&gt; representing the asynchronous operation.</returns>
        public async IAsyncEnumerable<VideoFrame> RunAsync(UpscaleStreamOptions options, IProgress<RunProgress> progressCallback = default, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var frameCount = 0;
            var timestamp = RunProgress.GetTimestamp();
            await foreach (var videoFrame in options.Stream)
            {
                var frameTime = Stopwatch.GetTimestamp();
                if (_upscaleModel.Normalization == Normalization.ZeroToOne)
                    videoFrame.Frame.NormalizeZeroToOne();

                var resultTensor = await UpscaleInternalAsync(videoFrame.Frame, options, cancellationToken);
                if (_upscaleModel.Normalization == Normalization.ZeroToOne)
                {
                    resultTensor.NormalizeOneToOne();
                    videoFrame.Frame.NormalizeOneToOne();
                }
                progressCallback?.Report(new RunProgress(++frameCount, 0, frameTime));
                yield return new VideoFrame(resultTensor, videoFrame.SourceFrameRate);
            }
            progressCallback?.Report(new RunProgress(timestamp));
        }


        /// <summary>
        /// Disposes this pipeline.
        /// </summary>
        public void Dispose()
        {
            _upscaleModel.Dispose();
        }


        /// <summary>
        /// Upscale ImageTensor with the specified UpscaleOptions
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        private async Task<ImageTensor> UpscaleInternalAsync(ImageTensor imageTensor, UpscaleOptions options, CancellationToken cancellationToken = default)
        {
            return options.TileMode == TileMode.None
                ? await ExecuteUpscaleAsync(imageTensor, cancellationToken)
                : await ExecuteUpscaleTilesAsync(imageTensor, options.MaxTileSize, options.TileMode, options.TileOverlap, cancellationToken);
        }


        /// <summary>
        /// Execute Upscaler
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        private async Task<ImageTensor> ExecuteUpscaleAsync(ImageTensor imageTensor, CancellationToken cancellationToken = default)
        {
            ThrowIfInvalidInput(imageTensor);
            var metadata = await _upscaleModel.LoadAsync(cancellationToken: cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            var outputDimension = new[] { 1, _upscaleModel.Channels, imageTensor.Height * _upscaleModel.ScaleFactor, imageTensor.Width * _upscaleModel.ScaleFactor };
            using (var modelParameters = new ModelParameters(metadata, cancellationToken))
            {
                modelParameters.AddInput(imageTensor.GetChannels(_upscaleModel.Channels));
                modelParameters.AddOutput(outputDimension);
                using (var results = await _upscaleModel.RunInferenceAsync(modelParameters))
                {
                    return results[0].ToTensor().AsImageTensor();
                }
            }
        }


        /// <summary>
        /// Execute Upscaler using tiles
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="maxTileSize">Maximum size of the tile.</param>
        /// <param name="tileOverlap">The tile overlap.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        private async Task<ImageTensor> ExecuteUpscaleTilesAsync(ImageTensor imageTensor, int maxTileSize, TileMode tileMode, int tileOverlap, CancellationToken cancellationToken = default)
        {
            if (_upscaleModel.SampleSize > 0)
                maxTileSize = _upscaleModel.SampleSize - tileOverlap;

            if (imageTensor.Width <= (maxTileSize + tileOverlap) || imageTensor.Height <= (maxTileSize + tileOverlap))
                return await ExecuteUpscaleAsync(imageTensor, cancellationToken);

            var inputTiles = new ImageTiles(imageTensor, tileMode, tileOverlap);
            var outputTiles = new ImageTiles
            (
                inputTiles.Width * _upscaleModel.ScaleFactor,
                inputTiles.Height * _upscaleModel.ScaleFactor,
                inputTiles.TileMode,
                inputTiles.Overlap * _upscaleModel.ScaleFactor,
                await ExecuteUpscaleTilesAsync(inputTiles.Tile1, maxTileSize, tileMode, tileOverlap, cancellationToken),
                await ExecuteUpscaleTilesAsync(inputTiles.Tile2, maxTileSize, tileMode, tileOverlap, cancellationToken),
                await ExecuteUpscaleTilesAsync(inputTiles.Tile3, maxTileSize, tileMode, tileOverlap, cancellationToken),
                await ExecuteUpscaleTilesAsync(inputTiles.Tile4, maxTileSize, tileMode, tileOverlap, cancellationToken)
            );
            return outputTiles.JoinTiles();
        }


        /// <summary>
        /// Throws exception if input is invalid.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        private void ThrowIfInvalidInput(ImageTensor imageTensor)
        {
            if (_upscaleModel.SampleSize > 0)
            {
                ArgumentOutOfRangeException.ThrowIfGreaterThan(imageTensor.Width, _upscaleModel.SampleSize, nameof(imageTensor.Width));
                ArgumentOutOfRangeException.ThrowIfGreaterThan(imageTensor.Height, _upscaleModel.SampleSize, nameof(imageTensor.Height));
            }
        }


        /// <summary>
        /// Creates an UpscalePipeline
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <returns>UpscalePipeline.</returns>
        public static UpscalePipeline Create(UpscalerConfig configuration)
        {
            var upscalerModel = UpscalerModel.Create(configuration);
            return new UpscalePipeline(upscalerModel);
        }

    }
}

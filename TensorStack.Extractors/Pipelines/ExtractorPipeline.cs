// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Image;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.Common.Video;
using TensorStack.Core;
using TensorStack.Core.Inference;
using TensorStack.Extractors.Common;
using TensorStack.Extractors.Models;

namespace TensorStack.Extractors.Pipelines
{
    /// <summary>
    /// Basic ExtractorPipeline. This class cannot be inherited.
    /// </summary>
    /// <seealso cref="TensorStack.Common.Pipeline.IPipelineImage{TensorStack.Extractors.Common.ExtractorOptions}" />
    /// <seealso cref="TensorStack.Common.Pipeline.IPipelineVideo{TensorStack.Extractors.Common.ExtractorOptions}" />
    public sealed class ExtractorPipeline : IPipelineImage<ExtractorOptions>, IPipelineVideo<ExtractorOptions>
    {
        private readonly ExtractorModel _extractorModel;

        /// <summary>
        /// Initializes a new instance of the <see cref="ExtractorPipeline"/> class.
        /// </summary>
        /// <param name="extractorModel">The extractor model.</param>
        public ExtractorPipeline(ExtractorModel extractorModel)
        {
            _extractorModel = extractorModel;
        }


        /// <summary>
        /// Loads the pipeline.
        /// </summary>
        public async Task LoadAsync()
        {
            await _extractorModel.LoadAsync();
        }

        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        public async Task UnloadAsync()
        {
            await _extractorModel.UnloadAsync();
        }


        /// <summary>
        /// Run the pipeline ImageTensor to ImageTensor function
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>Task&lt;ImageTensor&gt;.</returns>
        public Task<ImageTensor> RunImageAsync(ImageTensor inputImage, IProgress<RunProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            return RunImageAsync(default, inputImage, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Run the pipeline ImageTensor to ImageTensor function with the specified UpscaleOptions
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;ImageTensor&gt; representing the asynchronous operation.</returns>
        public async Task<ImageTensor> RunImageAsync(ExtractorOptions options, ImageTensor inputImage, IProgress<RunProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            options ??= new ExtractorOptions();
            var timestamp = RunProgress.GetTimestamp();
            if (_extractorModel.Normalization == Normalization.ZeroToOne)
                inputImage.NormalizeZeroToOne();

            var resultTensor = await ExtractInternalAsync(inputImage, options, cancellationToken);
            NormalizeResult(resultTensor);
            if (_extractorModel.Normalization == Normalization.ZeroToOne)
                inputImage.NormalizeOneToOne();
            if (options.MergeInput)
                resultTensor = MergeResult(inputImage, resultTensor);

            progressCallback?.Report(new RunProgress(timestamp));
            return resultTensor;
        }


        /// <summary>
        /// Run the pipeline VideoTensor to VideoTensor function
        /// </summary>
        /// <param name="inputImage">The input image.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>Task&lt;VideoTensor&gt;.</returns>
        public Task<VideoTensor> RunVideoAsync(VideoTensor inputImage, IProgress<RunProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            return RunVideoAsync(new ExtractorOptions(), inputImage, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Run the pipeline VideoTensor to VideoTensor function with the specified UpscaleOptions
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="inputVideo">The input video.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;VideoTensor&gt; representing the asynchronous operation.</returns>
        public async Task<VideoTensor> RunVideoAsync(ExtractorOptions options, VideoTensor inputVideo, IProgress<RunProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            options ??= new ExtractorOptions();
            var timestamp = RunProgress.GetTimestamp();
            if (_extractorModel.Normalization == Normalization.ZeroToOne)
                inputVideo.NormalizeZeroToOne();

            var results = new List<ImageTensor>();
            foreach (var frame in inputVideo.GetFrames())
            {
                var frameTime = Stopwatch.GetTimestamp();
                var resultTensor = await ExtractInternalAsync(frame, options, cancellationToken);
                NormalizeResult(resultTensor);
                if (_extractorModel.Normalization == Normalization.ZeroToOne)
                    frame.NormalizeOneToOne();
                if (options.MergeInput)
                    resultTensor = MergeResult(frame, resultTensor);

                results.Add(resultTensor);
                progressCallback?.Report(new RunProgress(results.Count, inputVideo.Frames, frameTime));
            }

            var resultVideoTensor = new VideoTensor(results.Join(), inputVideo.FrameRate);
            progressCallback?.Report(new RunProgress(timestamp));
            return resultVideoTensor;
        }


        /// <summary>
        /// Get the pipeline VideoFrame stream
        /// </summary>
        /// <param name="inputVideoStream">The input video stream.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;IAsyncEnumerable`1&gt; representing the asynchronous operation.</returns>
        public IAsyncEnumerable<VideoFrame> GetStreamAsync(IAsyncEnumerable<VideoFrame> inputVideoStream, IProgress<RunProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            return GetStreamAsync(new ExtractorOptions(), inputVideoStream, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Get the pipeline VideoFrame stream
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="inputVideoStream">The input video stream.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;IAsyncEnumerable`1&gt; representing the asynchronous operation.</returns>

        public async IAsyncEnumerable<VideoFrame> GetStreamAsync(ExtractorOptions options, IAsyncEnumerable<VideoFrame> inputVideoStream, IProgress<RunProgress> progressCallback = default, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var frameCount = 0;
            options ??= new ExtractorOptions();
            var timestamp = RunProgress.GetTimestamp();
            await foreach (var videoFrame in inputVideoStream)
            {
                var frameTime = Stopwatch.GetTimestamp();
                if (_extractorModel.Normalization == Normalization.ZeroToOne)
                    videoFrame.Frame.NormalizeZeroToOne();

                var resultTensor = await ExtractInternalAsync(videoFrame.Frame, options, cancellationToken);
                NormalizeResult(resultTensor);
                if (_extractorModel.Normalization == Normalization.ZeroToOne)
                    videoFrame.Frame.NormalizeOneToOne();
                if (options.MergeInput)
                    resultTensor = MergeResult(videoFrame.Frame, resultTensor);

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
            _extractorModel.Dispose();
        }


        /// <summary>
        /// Run Extractor on input ImageTensor with the specified UpscaleOptions
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        private async Task<ImageTensor> ExtractInternalAsync(ImageTensor imageTensor, ExtractorOptions options, CancellationToken cancellationToken = default)
        {
            return !options.TileMode
                ? await ExecuteExtractorAsync(imageTensor, cancellationToken)
                : await ExecuteExtractorTilesAsync(imageTensor, options.MaxTileSize, options.TileOverlap, cancellationToken);
        }


        /// <summary>
        /// Execute Extractor
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        private async Task<ImageTensor> ExecuteExtractorAsync(ImageTensor inputTensor, CancellationToken cancellationToken = default)
        {
            ThrowIfInvalidInput(inputTensor);
            var metadata = await _extractorModel.LoadAsync(cancellationToken: cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            var outputShape = new[] { 1, _extractorModel.OutputChannels, inputTensor.Dimensions[2], inputTensor.Dimensions[3] };
            var outputBuffer = metadata.Outputs[0].Value.Dimensions.Length == 4 ? outputShape : outputShape[1..];
            using (var inferenceParameters = new InferenceParameters(metadata, cancellationToken))
            {
                inferenceParameters.AddInput(inputTensor.GetChannels(_extractorModel.Channels));
                inferenceParameters.AddOutput(outputBuffer);
                using (var result = await _extractorModel.RunInferenceFirstAsync(inferenceParameters))
                {
                    var outputTensor = result.ToTensor();
                    if (outputBuffer.Length != 4)
                        outputTensor.Reshape(outputShape);

                    return outputTensor.AsImageTensor();
                }
            }
        }


        /// <summary>
        /// Execute Extractor using tiles
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="maxTileSize">Maximum size of the tile.</param>
        /// <param name="tileOverlap">The tile overlap.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;ImageTensor&gt; representing the asynchronous operation.</returns>
        private async Task<ImageTensor> ExecuteExtractorTilesAsync(ImageTensor imageTensor, int maxTileSize, int tileOverlap, CancellationToken cancellationToken = default)
        {
            if (_extractorModel.SampleSize > 0)
                maxTileSize = _extractorModel.SampleSize - tileOverlap;

            if (imageTensor.Width <= (maxTileSize + tileOverlap) || imageTensor.Height <= (maxTileSize + tileOverlap))
                return await ExecuteExtractorAsync(imageTensor, cancellationToken);

            var inputTiles = new ImageTiles(imageTensor, tileOverlap);
            var outputTiles = new ImageTiles
            (
                inputTiles.Width,
                inputTiles.Height,
                inputTiles.Overlap,
                await ExecuteExtractorTilesAsync(inputTiles.Tile1, maxTileSize, tileOverlap, cancellationToken),
                await ExecuteExtractorTilesAsync(inputTiles.Tile2, maxTileSize, tileOverlap, cancellationToken),
                await ExecuteExtractorTilesAsync(inputTiles.Tile3, maxTileSize, tileOverlap, cancellationToken),
                await ExecuteExtractorTilesAsync(inputTiles.Tile4, maxTileSize, tileOverlap, cancellationToken)
            );
            return outputTiles.JoinTiles();
        }


        /// <summary>
        /// Normalizes the output result.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        private void NormalizeResult(ImageTensor tensor)
        {
            if (_extractorModel.OutputNormalization == Normalization.MinMax)
                tensor.Memory.Span.NormalizeMinMaxToOneToOne();
            else if (_extractorModel.Normalization == Normalization.ZeroToOne)
                tensor.NormalizeOneToOne();

            if (_extractorModel.IsOutputInverted)
                tensor.Memory.Span.Invert();
        }


        /// <summary>
        /// Merges the input and output if required.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <returns>ImageTensor.</returns>
        private ImageTensor MergeResult(ImageTensor input, ImageTensor output)
        {
            var mergedInput = input.CloneAs();
            mergedInput.UpdateAlphaChannel(output);
            return mergedInput;
        }


        /// <summary>
        /// Throws exception if input is invalid.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        private void ThrowIfInvalidInput(ImageTensor imageTensor)
        {
            if (_extractorModel.SampleSize > 0)
            {
                ArgumentOutOfRangeException.ThrowIfGreaterThan(imageTensor.Width, _extractorModel.SampleSize, nameof(imageTensor.Width));
                ArgumentOutOfRangeException.ThrowIfGreaterThan(imageTensor.Height, _extractorModel.SampleSize, nameof(imageTensor.Height));
            }
        }


        /// <summary>
        /// Creates an ExtractorPipeline
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <returns>ExtractorPipeline.</returns>
        public static ExtractorPipeline Create(ExtractorConfig configuration)
        {
            var extractorModel = ExtractorModel.Create(configuration);
            return new ExtractorPipeline(extractorModel);
        }


        /// <summary>
        /// Creates an ExtractorPipeline
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="sessionOptionsFactory">The session options factory.</param>
        /// <returns>ExtractorPipeline.</returns>
        public static ExtractorPipeline Create(ExtractorConfig configuration, Func<SessionOptions> sessionOptionsFactory)
        {
            var extractorModel = ExtractorModel.Create(configuration, sessionOptionsFactory);
            return new ExtractorPipeline(extractorModel);
        }
    }
}

// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
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
using TensorStack.Extractors.Common;
using TensorStack.Extractors.Models;

namespace TensorStack.Extractors.Pipelines
{
    /// <summary>
    /// Basic ExtractorPipeline. This class cannot be inherited.
    /// </summary>
    public class ExtractorPipeline
        : IPipeline<ImageTensor, ExtractorImageOptions>,
          IPipeline<VideoTensor, ExtractorVideoOptions>,
          IPipelineStream<VideoFrame, ExtractorStreamOptions>
    {
        private readonly ExtractorModel _model;

        /// <summary>
        /// Initializes a new instance of the <see cref="ExtractorPipeline"/> class.
        /// </summary>
        /// <param name="extractorModel">The extractor model.</param>
        public ExtractorPipeline(ExtractorModel extractorModel)
        {
            _model = extractorModel;
        }


        /// <summary>
        /// Loads the pipeline.
        /// </summary>
        public async Task LoadAsync(CancellationToken cancellationToken = default)
        {
            await _model.LoadAsync(cancellationToken: cancellationToken);
        }

        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        public async Task UnloadAsync(CancellationToken cancellationToken = default)
        {
            await _model.UnloadAsync();
        }


        /// <summary>
        /// Run the pipeline ImageTensor to ImageTensor function with the specified UpscaleOptions
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;ImageTensor&gt; representing the asynchronous operation.</returns>
        public async Task<ImageTensor> RunAsync(ExtractorImageOptions options, IProgress<RunProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var timestamp = RunProgress.GetTimestamp();
            var metadata = await _model.LoadAsync(cancellationToken: cancellationToken);
            var resultTensor = await ExtractInternalAsync(metadata, options.Image, options, progressCallback, cancellationToken);

            if (options.MergeInput)
                resultTensor = MergeResult(options.Image, resultTensor);

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
        public async Task<VideoTensor> RunAsync(ExtractorVideoOptions options, IProgress<RunProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            var timestamp = RunProgress.GetTimestamp();
            var results = new List<ImageTensor>();
            var metadata = await _model.LoadAsync(cancellationToken: cancellationToken);
            foreach (var frame in options.Video.GetFrames())
            {
                var frameTime = Stopwatch.GetTimestamp();
                var resultTensor = await ExtractInternalAsync(metadata, frame, options, default, cancellationToken);
                if (options.MergeInput)
                    resultTensor = MergeResult(frame, resultTensor);

                results.Add(resultTensor);
                progressCallback?.Report(new RunProgress(results.Count, options.Video.Frames, frameTime));
            }

            var resultVideoTensor = new VideoTensor(results.Join(), options.Video.FrameRate);
            progressCallback?.Report(new RunProgress(timestamp));
            return resultVideoTensor;
        }


        /// <summary>
        /// Get the pipeline VideoFrame stream
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="inputVideoStream">The input video stream.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;IAsyncEnumerable`1&gt; representing the asynchronous operation.</returns>
        public async IAsyncEnumerable<VideoFrame> RunAsync(ExtractorStreamOptions options, IProgress<RunProgress> progressCallback = default, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var frameCount = 0;
            var timestamp = RunProgress.GetTimestamp();
            var metadata = await _model.LoadAsync(cancellationToken: cancellationToken);
            await foreach (var videoFrame in options.Stream)
            {
                var frameTime = Stopwatch.GetTimestamp();
                var resultTensor = await ExtractInternalAsync(metadata, videoFrame.Frame, options, default, cancellationToken);

                if (options.MergeInput)
                    resultTensor = MergeResult(videoFrame.Frame, resultTensor);

                progressCallback?.Report(new RunProgress(++frameCount, 0, frameTime));
                yield return new VideoFrame(videoFrame.Index, resultTensor, videoFrame.SourceFrameRate);
            }
            progressCallback?.Report(new RunProgress(timestamp));
        }


        /// <summary>
        /// Disposes this pipeline.
        /// </summary>
        public void Dispose()
        {
            _model.Dispose();
        }


        /// <summary>
        /// Run Extractor on input ImageTensor with the specified UpscaleOptions
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        private async Task<ImageTensor> ExtractInternalAsync(ModelMetadata modelMetadata, ImageTensor imageTensor, ExtractorOptions options, IProgress<RunProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();

            return options.IsTileEnabled == false
                ? await ExecuteExtractorAsync(modelMetadata, imageTensor, cancellationToken)
                : await ExecuteTileExtractorAsync(modelMetadata, imageTensor, options, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Execute Extractor infernece
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        private async Task<ImageTensor> ExecuteExtractorAsync(ModelMetadata modelMetadata, ImageTensor imageInput, CancellationToken cancellationToken = default)
        {
            // Resize Input
            var inputTensor = imageInput;
            if (_model.SampleSize > 0)
                inputTensor = inputTensor.ResizeImage(_model.SampleSize, _model.SampleSize, ResizeMode.Stretch);

            var outputShape = new[] { 1, _model.OutputChannels, inputTensor.Dimensions[2], inputTensor.Dimensions[3] };
            var outputBuffer = modelMetadata.Outputs[0].Value.Dimensions.Length == 4 ? outputShape : outputShape[1..];
            using (var modelParameters = new ModelParameters(modelMetadata, cancellationToken))
            {
                modelParameters.AddImageInput(inputTensor, _model.Channels, _model.Normalization);
                modelParameters.AddOutput(outputBuffer);

                var results = _model.IsDynamicOutput
                    ? _model.RunInference(modelParameters)
                    : await _model.RunInferenceAsync(modelParameters);
                using (results)
                {
                    // Output Tensor
                    var outputTensor = results[0].ToTensor();
                    if (outputBuffer.Length != 4)
                        outputTensor.Reshape([1, .. outputTensor.Dimensions]);

                    // Normalize Output
                    outputTensor.Normalize(_model.OutputNormalization);

                    // Resize Output
                    var outputImage = outputTensor.AsImageTensor();
                    if (_model.SampleSize > 0 || outputImage.Width != imageInput.Width || outputImage.Height != imageInput.Height)
                        outputImage = outputImage.ResizeImage(imageInput.Width, imageInput.Height, ResizeMode.Stretch);

                    return outputImage;
                }
            }
        }


        /// <summary>
        /// Execute tile extractor infernece.
        /// </summary>
        /// <param name="modelMetadata">The model metadata.</param>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="tileJob">The tile job.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<ImageTensor> ExecuteTileExtractorAsync(ModelMetadata modelMetadata, Tensor<float> imageTensor, TileJob tileJob, CancellationToken cancellationToken = default)
        {
            var tileTensor = ImageTiles.ExtractTileSpan(imageTensor, tileJob.X, tileJob.Y, tileJob.TileSize, _model.Channels);
            var tileHeight = tileTensor.Dimensions[2];
            var tileWidth = tileTensor.Dimensions[3];

            var outputShape = new[] { 1, _model.OutputChannels, tileHeight, tileWidth };
            var outputBuffer = modelMetadata.Outputs[0].Value.Dimensions.Length == 4 ? outputShape : outputShape[1..];
            using (var modelParameters = new ModelParameters(modelMetadata, cancellationToken))
            {
                modelParameters.AddInput(tileTensor, _model.Normalization);
                modelParameters.AddOutput(outputBuffer);

                var results = _model.IsDynamicOutput
                    ? _model.RunInference(modelParameters)
                    : await _model.RunInferenceAsync(modelParameters);
                using (results)
                {
                    // Output Tensor
                    var outputTensor = results[0].ToTensor();
                    if (outputBuffer.Length != 4)
                        outputTensor.Reshape([1, .. outputTensor.Dimensions]);

                    // Normalize Output
                    outputTensor.Normalize(_model.OutputNormalization);

                    // Resize Output
                    var outputImage = outputTensor.AsImageTensor();
                    if (outputImage.Width != tileWidth || outputImage.Height != tileHeight)
                        outputImage = outputImage.ResizeImage(tileWidth, tileHeight, ResizeMode.Stretch);

                    return outputImage;
                }
            }
        }


        /// <summary>
        /// Execute tiled extractor process
        /// </summary>
        /// <param name="metadata">The metadata.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<ImageTensor> ExecuteTileExtractorAsync(ModelMetadata metadata, ImageTensor inputImage, ExtractorOptions options, IProgress<RunProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            var scaleFactor = 1;
            if (inputImage.Width <= options.MaxTileSize && inputImage.Height <= options.MaxTileSize)
                return await ExecuteExtractorAsync(metadata, inputImage, cancellationToken);

            var tileSize = options.MaxTileSize - options.TileOverlap;
            var outputHeight = inputImage.Height * scaleFactor;
            var outputWidth = inputImage.Width * scaleFactor;
            var outputImage = new ImageTensor(outputHeight, outputWidth);

            var weightSum = ImageTiles.CreateWeightSum(outputImage);
            var imageTiles = ImageTiles.ComputeTiles(inputImage, tileSize, options.MaxTileSize);
            var weightMap = ImageTiles.CreateWeightMap(options.MaxTileSize, options.TileOverlap);
            for (int i = 0; i < imageTiles.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var imageTile = imageTiles[i];
                var upscaledTile = await ExecuteTileExtractorAsync(metadata, inputImage, imageTile, cancellationToken);
                ImageTiles.BlendTile(outputImage, weightSum, upscaledTile, weightMap, imageTile.X, imageTile.Y, scaleFactor);

                progressCallback?.Report(new RunProgress(i + 1, imageTiles.Count));
            }
            return ImageTiles.Normalize(outputImage, weightSum);
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
        /// Creates an ExtractorPipeline
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <returns>ExtractorPipeline.</returns>
        public static ExtractorPipeline Create(ExtractorConfig configuration)
        {
            return new ExtractorPipeline(ExtractorModel.Create(configuration));
        }

    }
}

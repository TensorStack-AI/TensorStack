// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.Extractors.Common;
using TensorStack.Extractors.Models;

namespace TensorStack.Extractors.Pipelines
{
    /// <summary>
    /// Basic BackgroundPipeline. This class cannot be inherited.
    /// </summary>
    public class BackgroundPipeline
        : IPipeline<ImageTensor, BackgroundImageOptions>
    {
        private readonly ExtractorModel _model;

        /// <summary>
        /// Initializes a new instance of the <see cref="BackgroundPipeline"/> class.
        /// </summary>
        /// <param name="backgroundModel">The background model.</param>
        public BackgroundPipeline(ExtractorModel backgroundModel)
        {
            _model = backgroundModel;
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
        /// Run the pipeline ImageTensor to ImageTensor function with the specified BackgroundImageOptions
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="inputImage">The input image.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;ImageTensor&gt; representing the asynchronous operation.</returns>
        public async Task<ImageTensor> RunAsync(BackgroundImageOptions options, IProgress<RunProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var timestamp = RunProgress.GetTimestamp();
            var resultTensor = await ExtractBackgroundInternalAsync(options, cancellationToken);
            progressCallback?.Report(new RunProgress(timestamp));
            return resultTensor;
        }


        /// <summary>
        /// Disposes this pipeline.
        /// </summary>
        public void Dispose()
        {
            _model.Dispose();
        }


        /// <summary>
        /// Run Extract Background on input ImageTensor
        /// </summary>
        /// <param name="imageInput">The image tensor.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        private async Task<ImageTensor> ExtractBackgroundInternalAsync(BackgroundImageOptions options, CancellationToken cancellationToken = default)
        {
            var metadata = await _model.LoadAsync(cancellationToken: cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();

            // Resize Input
            var inputTensor = options.Image;
            var sampleSize = _model.SampleSize;
            if (inputTensor.Width != sampleSize || inputTensor.Height != sampleSize)
                inputTensor = inputTensor.ResizeImage(sampleSize, sampleSize, ResizeMode.Stretch, ResizeMethod.Bicubic);

            var outputShape = new[] { 1, _model.OutputChannels, inputTensor.Dimensions[2], inputTensor.Dimensions[3] };
            var outputBuffer = metadata.Outputs[0].Value.Dimensions.Length == 4 ? outputShape : outputShape[1..];
            using (var modelParameters = new ModelParameters(metadata, cancellationToken))
            {
                modelParameters.AddImageInput(inputTensor, _model.Channels, _model.Normalization);
                modelParameters.AddOutput(outputBuffer);
                using (var results = await _model.RunInferenceAsync(modelParameters))
                {
                    // Output Tensor
                    var outputTensor = results[0].ToTensor();
                    if (outputBuffer.Length != 4)
                        outputTensor.Reshape([1, .. outputTensor.Dimensions]);
       
                    // Normalize
                    outputTensor.Normalize(_model.OutputNormalization);

                    // Process Image
                    var outputImage = default(ImageTensor);
                    if (options.Mode == BackgroundMode.MaskForeground || options.Mode == BackgroundMode.MaskBackground)
                    {
                        if (options.Mode == BackgroundMode.MaskBackground)
                            outputTensor.Invert();
                        outputImage = new ImageTensor(inputTensor.Height, inputTensor.Width, options.MaskFill);
                    }
                    else if (options.Mode == BackgroundMode.RemoveBackground || options.Mode == BackgroundMode.RemoveForeground)
                    {
                        if (options.Mode == BackgroundMode.RemoveForeground)
                            outputTensor.Invert();
                        outputImage = inputTensor.CloneAs();
                    }

                    // Set Alpha Channel
                    if (options.IsTransparentSupported)
                    {
                        outputImage.UpdateAlphaChannel(outputTensor.Span);
                    }
                    else
                    {
                        outputImage.FlattenAlphaChannel(outputTensor.Span);
                    }

                    // Resize Output
                    if (outputImage.Width != options.Image.Width || outputImage.Height != options.Image.Height)
                        outputImage.Resize(options.Image.Width, options.Image.Height, ResizeMode.Stretch, ResizeMethod.Bilinear);

                    return outputImage;
                }
            }
        }


        /// <summary>
        /// Creates an BackgroundPipeline
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <returns>BackgroundPipeline.</returns>
        public static BackgroundPipeline Create(ExtractorConfig configuration)
        {
            return new BackgroundPipeline(ExtractorModel.Create(configuration));
        }
    }
}

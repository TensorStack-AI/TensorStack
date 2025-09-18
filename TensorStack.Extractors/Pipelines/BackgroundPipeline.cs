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
            if (_model.Normalization == Normalization.ZeroToOne)
                options.Input.NormalizeZeroToOne();

            var resultTensor = await ExtractBackgroundInternalAsync(options.Input, cancellationToken);
            NormalizeOutput(resultTensor);

            if (_model.Normalization == Normalization.ZeroToOne)
                options.Input.NormalizeOneToOne();

            if (options.Mode == BackgroundMode.RemoveForeground || options.Mode == BackgroundMode.MaskBackground)
                resultTensor.Memory.Span.Invert();

            if (options.Mode == BackgroundMode.RemoveBackground || options.Mode == BackgroundMode.RemoveForeground)
                resultTensor = AddAlphaChannel(options.Input, resultTensor);

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
        private async Task<ImageTensor> ExtractBackgroundInternalAsync(ImageTensor imageInput, CancellationToken cancellationToken = default)
        {
            // Resize Input
            var inputTensor = imageInput;
            var sampleSize = _model.SampleSize;
            if (inputTensor.Width != sampleSize || inputTensor.Height != sampleSize)
                inputTensor = inputTensor.ResizeImage(sampleSize, sampleSize, ResizeMode.Stretch);

            var metadata = await _model.LoadAsync(cancellationToken: cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();

            var outputShape = new[] { 1, _model.OutputChannels, inputTensor.Dimensions[2], inputTensor.Dimensions[3] };
            var outputBuffer = metadata.Outputs[0].Value.Dimensions.Length == 4 ? outputShape : outputShape[1..];
            using (var modelParameters = new ModelParameters(metadata, cancellationToken))
            {
                modelParameters.AddInput(inputTensor.GetChannels(_model.Channels));
                modelParameters.AddOutput(outputBuffer);
                using (var results = await _model.RunInferenceAsync(modelParameters))
                {
                    var outputTensor = results[0].ToTensor();
                    if (outputBuffer.Length != 4)
                        outputTensor.Reshape([1, .. outputTensor.Dimensions]);

                    // Resize Output
                    var outputImage = outputTensor.AsImageTensor();
                    if (outputImage.Width != imageInput.Width || outputImage.Height != imageInput.Height)
                        outputImage = outputImage.ResizeImage(imageInput.Width, imageInput.Height, ResizeMode.Stretch);

                    return outputImage;
                }
            }
        }


        /// <summary>
        /// Merges the input and output if required.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <returns>ImageTensor.</returns>
        private ImageTensor AddAlphaChannel(ImageTensor input, ImageTensor output)
        {
            var mergedInput = input.CloneAs();
            mergedInput.UpdateAlphaChannel(output);
            return mergedInput;
        }


        /// <summary>
        /// Normalizes the output.
        /// </summary>
        /// <param name="resultTensor">The result tensor.</param>
        private void NormalizeOutput(ImageTensor resultTensor)
        {
            if (_model.OutputNormalization == Normalization.OneToOne)
                resultTensor.NormalizeOneToOne();
            else if (_model.OutputNormalization == Normalization.ZeroToOne)
                resultTensor.NormalizeZeroToOne();
            else if (_model.OutputNormalization == Normalization.MinMax)
                resultTensor.Memory.Span.NormalizeMinMaxToZeroToOne();
        }


        /// <summary>
        /// Creates an BackgroundPipeline
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <returns>BackgroundPipeline.</returns>
        public static BackgroundPipeline Create(ExtractorConfig configuration)
        {
            var extractorModel = ExtractorModel.Create(configuration);
            return new BackgroundPipeline(extractorModel);
        }
    }
}

// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Config;

namespace TensorStack.StableDiffusion.Models
{
    public class AutoEncoderModel : IDisposable
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AutoEncoderModel"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public AutoEncoderModel(AutoEncoderModelConfig configuration)
        {
            Configuration = configuration;
            Decoder = new ModelSession<AutoEncoderModelConfig>(Configuration with { Path = Configuration.DecoderModelPath });
            if (!string.IsNullOrEmpty(Configuration.EncoderModelPath))
                Encoder = new ModelSession<AutoEncoderModelConfig>(Configuration with { Path = Configuration.EncoderModelPath });
        }

        /// <summary>
        /// Gets the configuration.
        /// </summary>
        protected AutoEncoderModelConfig Configuration { get; }

        /// <summary>
        /// Gets the Decoder model.
        /// </summary>
        protected ModelSession<AutoEncoderModelConfig> Decoder { get; }

        /// <summary>
        /// Gets the Encoder model.
        /// </summary>
        protected ModelSession<AutoEncoderModelConfig> Encoder { get; }

        /// <summary>
        /// Gets the scale.
        /// </summary>
        public int Scale => Configuration.Scale;

        /// <summary>
        /// Gets the input channels.
        /// </summary>
        public int InChannels => Configuration.InChannels;

        /// <summary>
        /// Gets the output channels.
        /// </summary>
        public int OutChannels => Configuration.OutChannels;

        /// <summary>
        /// Gets the latent channels.
        /// </summary>
        /// <value>The latent channels.</value>
        public int LatentChannels => Configuration.LatentChannels;

        /// <summary>
        /// Gets the scale factor.
        /// </summary>
        public float ScaleFactor => Configuration.ScaleFactor;

        /// <summary>
        /// Gets the shift factor.
        /// </summary>
        public float ShiftFactor => Configuration.ShiftFactor;


        /// <summary>
        /// Indicates if the Decoder is loaded
        /// </summary>
        public bool IsDecoderLoaded()
        {
            return Decoder.IsLoaded();
        }


        /// <summary>
        /// Indicates if the Encoder is loaded
        /// </summary>
        public bool IsEncoderLoaded()
        {
            return Encoder.IsLoaded();
        }


        /// <summary>
        /// Loads the Decoder model.
        /// </summary>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<ModelMetadata> DecoderLoadAsync(CancellationToken cancellationToken)
        {
            return await Decoder.LoadAsync(cancellationToken: cancellationToken);
        }


        /// <summary>
        /// Loads the Encoder model.
        /// </summary>
        /// <param name="cancellationToken">The cancellation token.</para
        public async Task<ModelMetadata> EncoderLoadAsync(CancellationToken cancellationToken)
        {
            return await Encoder.LoadAsync(cancellationToken: cancellationToken);
        }


        /// <summary>
        /// Unloads the Decoder model.
        /// </summary>
        public async Task DecoderUnloadAsync()
        {
            await Decoder?.UnloadAsync();
        }


        /// <summary>
        /// Unloads the Encoder model.
        /// </summary>
        public async Task EncoderUnloadAsync()
        {
            await Encoder?.UnloadAsync();
        }


        /// <summary>
        /// Runs the Decoder model with the specified input
        /// </summary>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public virtual async Task<Tensor<float>> DecodeAsync(Tensor<float> inputTensor, bool disableShift = false, bool disableScale = false, CancellationToken cancellationToken = default)
        {
            if (!IsDecoderLoaded())
                await Decoder.LoadAsync(cancellationToken: cancellationToken);

            if (!disableScale)
                inputTensor.Multiply(1.0f / ScaleFactor);
            if (!disableShift)
                inputTensor.Add(ShiftFactor);

            var outputDimensions = new[] { 1, OutChannels, inputTensor.Dimensions[2] * Scale, inputTensor.Dimensions[3] * Scale };
            using (var modelParameters = new ModelParameters(Decoder.Metadata, cancellationToken))
            {
                // Inputs
                modelParameters.AddInput(inputTensor.AsTensorSpan());

                // Outputs
                modelParameters.AddOutput(outputDimensions);

                // Inference
                using (var results = await Decoder.RunInferenceAsync(modelParameters))
                {
                    return results[0].ToTensor();
                }
            }
        }


        /// <summary>
        /// Runs the Encoder model with the specified input
        /// </summary>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public virtual async Task<Tensor<float>> EncodeAsync(ImageTensor inputTensor, bool disableShift = false, bool disableScale = false, CancellationToken cancellationToken = default)
        {
            if (!IsEncoderLoaded())
                await Encoder.LoadAsync(cancellationToken: cancellationToken);

            var outputDimensions = new[] { 1, LatentChannels, inputTensor.Dimensions[2] / Scale, inputTensor.Dimensions[3] / Scale };
            using (var modelParameters = new ModelParameters(Encoder.Metadata, cancellationToken))
            {
                // Inputs
                modelParameters.AddInput(inputTensor.GetChannels(3));

                // Outputs
                modelParameters.AddOutput(outputDimensions);

                // Inference
                var results = await Encoder.RunInferenceAsync(modelParameters);
                using (var result = results.First())
                {
                    var tensorResult = result.ToTensor();
                    if (!disableShift)
                        tensorResult.Subtract(ShiftFactor);
                    if (!disableScale)
                        tensorResult.Multiply(ScaleFactor);

                    return tensorResult;
                }
            }
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Decoder?.Dispose();
            Encoder?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}

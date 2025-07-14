// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Config;
using TensorStack.StableDiffusion.Enums;

namespace TensorStack.StableDiffusion.Models
{
    /// <summary>
    /// UNetConditionalModel: Conditional U-Net architecture to denoise the encoded image latents.
    /// </summary>
    public class UNetConditionalModel : IDisposable
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="UNetConditionalModel"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public UNetConditionalModel(UNetModelConfig configuration)
        {
            Configuration = configuration;
            Unet = new ModelSession<UNetModelConfig>(configuration);
            if (!string.IsNullOrEmpty(configuration.ControlNetPath))
                UnetControlNet = new ModelSession<UNetModelConfig>(configuration with { Path = configuration.ControlNetPath });
        }

        /// <summary>
        /// Gets the configuration.
        /// </summary>
        protected UNetModelConfig Configuration { get; }

        /// <summary>
        /// Gets the Unet.
        /// </summary>
        protected ModelSession<UNetModelConfig> Unet { get; }

        /// <summary>
        /// Gets the unet control Unet ControlNet.
        /// </summary>
        protected ModelSession<UNetModelConfig> UnetControlNet { get; }

        /// <summary>
        /// Gets the in channels.
        /// </summary>
        public int InChannels => Configuration.InChannels;

        /// <summary>
        /// Gets the out channels.
        /// </summary>
        public int OutChannels => Configuration.OutChannels;

        /// <summary>
        /// Gets the type of the model.
        /// </summary>
        public ModelType ModelType => Configuration.ModelType;

        /// <summary>
        /// Gets a value indicating whether this instance has ControlNet.
        /// </summary>
        public bool HasControlNet => UnetControlNet != null;


        /// <summary>
        /// Determines whether Unet is loaded.
        /// </summary>
        public bool IsLoaded()
        {
            return Unet.IsLoaded();
        }


        /// <summary>
        /// Determines whether UnetControlNet is loaded.
        /// </summary>
        public bool IsControlNetLoaded()
        {
            return UnetControlNet.IsLoaded();
        }


        /// <summary>
        /// Load Unet model
        /// </summary>
        /// <param name="onnxOptimizations">The onnx optimizations.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<ModelMetadata> LoadAsync(ModelOptimization onnxOptimizations = null, CancellationToken cancellationToken = default)
        {
            await UnloadControlNetAsync();
            return await Unet.LoadAsync(onnxOptimizations, cancellationToken);
        }


        /// <summary>
        /// Load UnetControlNet model
        /// </summary>
        /// <param name="onnxOptimizations">The onnx optimizations.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<ModelMetadata> LoadControlNetAsync(ModelOptimization onnxOptimizations = null, CancellationToken cancellationToken = default)
        {
            await UnloadAsync();
            return await UnetControlNet.LoadAsync(onnxOptimizations, cancellationToken);
        }


        /// <summary>
        /// Unload Unet model
        /// </summary>
        /// <returns>A Task representing the asynchronous operation.</returns>
        public async Task UnloadAsync()
        {
            if (Unet.IsLoaded())
                await Unet.UnloadAsync();
        }


        /// <summary>
        /// Unload UnetControlNet model
        /// </summary>
        public async Task UnloadControlNetAsync()
        {
            if (UnetControlNet.IsLoaded())
                await UnetControlNet.UnloadAsync();
        }


        /// <summary>
        /// Determines whether Unet optimizations have changed
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        public bool HasOptimizationsChanged(ModelOptimization optimizations)
        {
            return Unet.HasOptimizationsChanged(optimizations);
        }


        /// <summary>
        /// Determines whether ControlNet optimizations have changed
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        public bool HasControlNetOptimizationsChanged(ModelOptimization optimizations)
        {
            return UnetControlNet.HasOptimizationsChanged(optimizations);
        }


        /// <summary>
        /// Runs the Unet model with the specified inputs
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        /// <param name="encoderHiddenStates">The encoder hidden states.</param>
        /// <param name="textEmbeds">The text embeds.</param>
        /// <param name="timeIds">The time ids.</param>
        /// <param name="outputDimension">The output dimension.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<Tensor<float>> RunAsync(int timestep, Tensor<float> sample, Tensor<float> encoderHiddenStates, Tensor<float> textEmbeds = default, Tensor<float> timeIds = default, CancellationToken cancellationToken = default)
        {
            if (UnetControlNet.IsLoaded())
                await UnetControlNet.UnloadAsync();
            if (!Unet.IsLoaded())
                await Unet.LoadAsync(cancellationToken: cancellationToken);

            var isStableDiffusionXL = Unet.Metadata.Inputs.Count == 5;
            var outputDimension = new[] { sample.Dimensions[0], OutChannels, sample.Dimensions[2], sample.Dimensions[3] };
            using (var modelParameters = new ModelParameters(Unet.Metadata, cancellationToken))
            {
                // Inputs
                modelParameters.AddInput(sample.AsTensorSpan());
                modelParameters.AddScalarInput(timestep);
                modelParameters.AddInput(encoderHiddenStates.AsTensorSpan());
                if (isStableDiffusionXL)
                {
                    modelParameters.AddInput(textEmbeds.AsTensorSpan());
                    modelParameters.AddInput(timeIds.AsTensorSpan());
                }

                // Outputs
                modelParameters.AddOutput(outputDimension);

                // Inference
                using (var results = await Unet.RunInferenceAsync(modelParameters))
                {
                    return results[0].ToTensor();
                }
            }
        }


        /// <summary>
        /// Runs the Unet + ControlNet model with the specified inputs
        /// </summary>
        /// <param name="controlNet">The control net.</param>
        /// <param name="controlImage">The control image.</param>
        /// <param name="conditioningScale">The conditioning scale.</param>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        /// <param name="encoderHiddenStates">The encoder hidden states.</param>
        /// <param name="textEmbeds">The text embeds.</param>
        /// <param name="timeIds">The time ids.</param>
        /// <param name="controlSample">The control sample.</param>
        /// <param name="outputDimension">The output dimension.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<Tensor<float>> RunAsync(ControlNetModel controlNet, ImageTensor controlImage, float conditioningScale, int timestep, Tensor<float> sample, Tensor<float> encoderHiddenStates, Tensor<float> textEmbeds = default, Tensor<float> timeIds = default, CancellationToken cancellationToken = default)
        {
            if (Unet.IsLoaded())
                await Unet.UnloadAsync();
            if (!UnetControlNet.IsLoaded())
                await UnetControlNet.LoadAsync(cancellationToken: cancellationToken);
            if (!controlNet.IsLoaded())
                await controlNet.LoadAsync(cancellationToken: cancellationToken);

            var isStableDiffusionXL = controlNet.Metadata.Inputs.Count == 7;
            var outputDimension = new[] { sample.Dimensions[0], OutChannels, sample.Dimensions[2], sample.Dimensions[3] };
            using (var unetParameters = new ModelParameters(UnetControlNet.Metadata, cancellationToken))
            using (var controlNetParameters = new ModelParameters(controlNet.Metadata, cancellationToken))
            {
                // Inputs ControlNet
                controlNetParameters.AddInput(sample.AsTensorSpan());
                controlNetParameters.AddScalarInput(timestep);
                controlNetParameters.AddInput(encoderHiddenStates.AsTensorSpan());
                if (isStableDiffusionXL)
                {
                    controlNetParameters.AddInput(textEmbeds.AsTensorSpan());
                    controlNetParameters.AddInput(timeIds.AsTensorSpan());
                }
                controlNetParameters.AddInput(controlImage.GetChannels(3));
                controlNetParameters.AddScalarInput(conditioningScale);

                // Outputs ControlNet
                foreach (var item in controlNet.Metadata.Outputs)
                    controlNetParameters.AddOutput();

                // ControlNet inference
                var controlNetResults = controlNet.RunInference(controlNetParameters);

                // Inputs Unet
                unetParameters.AddInput(sample.AsTensorSpan());
                unetParameters.AddScalarInput(timestep);
                unetParameters.AddInput(encoderHiddenStates.AsTensorSpan());
                if (isStableDiffusionXL)
                {
                    unetParameters.AddInput(textEmbeds.AsTensorSpan());
                    unetParameters.AddInput(timeIds.AsTensorSpan());
                }
                foreach (var item in controlNetResults)
                    unetParameters.AddInput(item);

                // Outputs Unet
                unetParameters.AddOutput(outputDimension);

                // Inference

                using (var results = await UnetControlNet.RunInferenceAsync(unetParameters))
                {
                    return results[0].ToTensor();
                }
            }
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Unet?.Dispose();
            UnetControlNet?.Dispose();
            GC.SuppressFinalize(this);
        }

    }
}

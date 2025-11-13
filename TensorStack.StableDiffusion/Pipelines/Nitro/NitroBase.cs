// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Enums;
using TensorStack.StableDiffusion.Models;
using TensorStack.StableDiffusion.Schedulers;
using TensorStack.TextGeneration.Pipelines.Llama;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.StableDiffusion.Pipelines.Nitro
{
    public abstract class NitroBase : PipelineBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="NitroBase"/> class.
        /// </summary>
        /// <param name="transformer">The transformer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="autoEncoder">The automatic encoder.</param>
        /// <param name="logger">The logger.</param>
        public NitroBase(TransformerNitroModel transformer, LlamaPipeline textEncoder, AutoEncoderModel autoEncoder, ILogger logger = default) : base(logger)
        {
            Transformer = transformer;
            AutoEncoder = autoEncoder;
            TextEncoder = textEncoder;
            Initialize();
            Logger?.LogInformation("[NitroPipeline] Name: {Name}", Name);
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="NitroBase"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public NitroBase(NitroConfig configuration, ILogger logger = default) : this(
            new TransformerNitroModel(configuration.Transformer),
            new LlamaPipeline(new LlamaConfig
            {
                OutputLastHiddenStates = true,
                DecoderConfig = configuration.TextEncoder,
                Tokenizer = new BPETokenizer(configuration.Tokenizer),
            }),
            new AutoEncoderModel(configuration.AutoEncoder),
            logger)
        {
            Name = configuration.Name;
        }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.Nitro;

        /// <summary>
        /// Gets the friendly name.
        /// </summary>
        public override string Name { get; init; } = nameof(PipelineType.Nitro);

        /// <summary>
        /// Gets the TextEncoder.
        /// </summary>
        public LlamaPipeline TextEncoder { get; init; }

        /// <summary>
        /// Gets the transformer.
        /// </summary>
        public TransformerNitroModel Transformer { get; init; }

        /// <summary>
        /// Gets the automatic encoder.
        /// </summary>
        public AutoEncoderModel AutoEncoder { get; init; }


        /// <summary>
        /// Loads the pipeline.
        /// </summary>
        /// <param name="cancellationToken">The cancellation token.</param>
        public Task LoadAsync(CancellationToken cancellationToken = default)
        {
            // Nitro pipelines are lazy loaded on first run
            return Task.CompletedTask;
        }


        /// <summary>
        /// Unloads the pipeline.
        /// </summary>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task UnloadAsync(CancellationToken cancellationToken = default)
        {
            await Task.WhenAll
            (
                Transformer.UnloadAsync(),
                TextEncoder.UnloadAsync(cancellationToken),
                AutoEncoder.EncoderUnloadAsync(),
                AutoEncoder.DecoderUnloadAsync()
            );
            Logger?.LogInformation("[{PipeLineType}] Pipeline Unloaded", PipelineType);
        }


        /// <summary>
        /// Validates the options.
        /// </summary>
        /// <param name="options">The options.</param>
        protected override void ValidateOptions(GenerateOptions options)
        {
            base.ValidateOptions(options);
            if (!Transformer.HasControlNet && options.HasControlNet)
                throw new ArgumentException("Model does not support ControlNet");
        }


        /// <summary>
        /// Creates the prompt input embeddings.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected async Task<PromptResult> CreatePromptAsync(IPipelineOptions options, CancellationToken cancellationToken = default)
        {
            // Conditional Prompt
            var promptEmbeds = await TextEncoder.GetLastHiddenState(new TextGeneration.Common.GenerateOptions
            {
                Seed = options.Seed,
                Prompt = options.Prompt,
                MinLength = 128,
                MaxLength = 128
            }, cancellationToken);

            // Unconditional prompt
            var negativePromptEmbeds = default(Tensor<float>);
            if (Transformer.ModelType != ModelType.Turbo)
            {
                negativePromptEmbeds = await TextEncoder.GetLastHiddenState(new TextGeneration.Common.GenerateOptions
                {
                    Seed = options.Seed,
                    Prompt = options.NegativePrompt,
                    MinLength = 128,
                    MaxLength = 128
                }, cancellationToken);
            }

            return new PromptResult(promptEmbeds, default, negativePromptEmbeds, default);
        }


        /// <summary>
        /// Decode the model latents to image
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="latents">The latents.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected async Task<ImageTensor> DecodeLatentsAsync(IPipelineOptions options, Tensor<float> latents, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[DecodeLatentsAsync] Begin AutoEncoder Decode");
            var decoderResult = await AutoEncoder.DecodeAsync(latents, cancellationToken: cancellationToken);
            if (options.IsLowMemoryEnabled || options.IsLowMemoryDecoderEnabled)
                await AutoEncoder.DecoderUnloadAsync();

            Logger.LogEnd(LogLevel.Debug, timestamp, "[DecodeLatentsAsync] AutoEncoder Decode Complete");
            return decoderResult.AsImageTensor();
        }


        /// <summary>
        /// Encode the image to model latents
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="image">The latents.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<Tensor<float>> EncodeLatentsAsync(IPipelineOptions options, ImageTensor image, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[EncodeLatentsAsync] Begin AutoEncoder Encode");
            var inputTensor = image.ResizeImage(options.Width, options.Height);
            var encoderResult = await AutoEncoder.EncodeAsync(inputTensor, cancellationToken: cancellationToken);
            if (options.IsLowMemoryEnabled || options.IsLowMemoryEncoderEnabled)
                await AutoEncoder.EncoderUnloadAsync();

            Logger.LogEnd(LogLevel.Debug, timestamp, "[EncodeLatentsAsync] AutoEncoder Encode Complete");
            return encoderResult;
        }


        protected async Task<Tensor<float>> RunInferenceAsync(IPipelineOptions options, IScheduler scheduler, PromptResult prompt, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[RunInferenceAsync] Begin Transformer Inference");

            // Prompt
            var isGuidanceEnabled = IsGuidanceEnabled(options);
            var promptEmbedsCond = prompt.PromptEmbeds;
            var promptEmbedsUncond = prompt.NegativePromptEmbeds;

            // Latents
            var latents = await CreateLatentInputAsync(options, scheduler, cancellationToken);

            // Load Model
            await LoadTransformerAsync(options, progressCallback, cancellationToken);

            // Timesteps
            var timesteps = scheduler.GetTimesteps();
            for (int i = 0; i < timesteps.Count; i++)
            {
                var timestep = timesteps[i];
                var steptime = Stopwatch.GetTimestamp();
                cancellationToken.ThrowIfCancellationRequested();

                // Inputs.
                var latentInput = scheduler.ScaleInput(timestep, latents);

                // Inference
                var conditional = await Transformer.RunAsync(timestep, latentInput, promptEmbedsCond, cancellationToken: cancellationToken);
                if (isGuidanceEnabled)
                {
                    var unconditional = await Transformer.RunAsync(timestep, latentInput, promptEmbedsUncond, cancellationToken: cancellationToken);
                    conditional = ApplyGuidance(conditional, unconditional, options.GuidanceScale);
                }

                // Scheduler
                var stepResult = scheduler.Step(timestep, conditional, latents);

                // Result
                latents = stepResult.Sample;

                // Progress
                if (scheduler.IsFinalOrder)
                    progressCallback.Notify(scheduler.CurrentStep, scheduler.TotalSteps, latents, steptime);

                Logger.LogEnd(LogLevel.Debug, steptime, $"[RunInferenceAsync] Step: {i + 1}/{timesteps.Count}");
            }

            // Unload
            if (options.IsLowMemoryEnabled || options.IsLowMemoryComputeEnabled)
                await Transformer.UnloadAsync();

            Logger.LogEnd(LogLevel.Debug, timestamp, "[RunInferenceAsync] Transformer Inference Complete");
            return latents;
        }


        /// <summary>
        /// Create latent input.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<Tensor<float>> CreateLatentInputAsync(IPipelineOptions options, IScheduler scheduler, CancellationToken cancellationToken = default)
        {
            var dimensions = new int[] { 1, AutoEncoder.LatentChannels, options.Height / AutoEncoder.Scale, options.Width / AutoEncoder.Scale };
            var noiseTensor = scheduler.CreateRandomSample(dimensions);
            if (options.HasInputImage)
            {
                var timestep = scheduler.GetStartTimestep();
                var encoderResult = await EncodeLatentsAsync(options, options.InputImage, cancellationToken);
                return scheduler.ScaleNoise(timestep, encoderResult, noiseTensor);
            }
            return noiseTensor;
        }


        /// <summary>
        /// Gets the model optimizations.
        /// </summary>
        /// <param name="generateOptions">The generate options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        private ModelOptimization GetOptimizations(IPipelineOptions generateOptions, IProgress<GenerateProgress> progressCallback = null)
        {
            var optimizations = new ModelOptimization(Optimization.None);
            if (Transformer.HasOptimizationsChanged(optimizations))
            {
                progressCallback.Notify("Optimizing Pipeline...");
            }
            return optimizations;
        }


        /// <summary>
        /// Determines whether classifier-free guidance is enabled
        /// </summary>
        /// <param name="options">The options.</param>
        private bool IsGuidanceEnabled(IPipelineOptions options)
        {
            return options.GuidanceScale > 1;
        }


        /// <summary>
        /// Load Transformer with optimizations
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<ModelMetadata> LoadTransformerAsync(IPipelineOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var optimizations = GetOptimizations(options, progressCallback);
            return await Transformer.LoadAsync(optimizations, cancellationToken);
        }


        /// <summary>
        /// Checks the state of the pipeline.
        /// </summary>
        /// <param name="options">The options.</param>
        protected override async Task CheckPipelineState(IPipelineOptions options)
        {
            // Check Transformer/ControlNet status
            if (options.HasControlNet && Transformer.IsLoaded())
                await Transformer.UnloadAsync();
            if (!options.HasControlNet && Transformer.IsControlNetLoaded())
                await Transformer.UnloadControlNetAsync();

            // Check LowMemory status
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryTextEncoderEnabled)) // TODO
                await TextEncoder.UnloadAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryComputeEnabled) && Transformer.IsLoaded())
                await Transformer.UnloadAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryComputeEnabled) && Transformer.IsControlNetLoaded())
                await Transformer.UnloadControlNetAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryEncoderEnabled) && AutoEncoder.IsEncoderLoaded())
                await AutoEncoder.EncoderUnloadAsync();
            if ((options.IsLowMemoryEnabled || options.IsLowMemoryDecoderEnabled) && AutoEncoder.IsDecoderLoaded())
                await AutoEncoder.DecoderUnloadAsync();
        }


        /// <summary>
        /// Configures the supported schedulers.
        /// </summary>
        protected override IReadOnlyList<SchedulerType> ConfigureSchedulers()
        {
            return [SchedulerType.FlowMatchEulerDiscrete, SchedulerType.FlowMatchEulerDynamic];
        }


        /// <summary>
        /// Configures the default SchedulerOptions.
        /// </summary>
        protected override GenerateOptions ConfigureDefaultOptions()
        {
            var options = new GenerateOptions
            {
                Steps = 20,
                Shift = 1f,
                Width = 512,
                Height = 512,
                GuidanceScale = 4f,
                Scheduler = SchedulerType.FlowMatchEulerDiscrete
            };

            // Nitro-Distilled Models ,4 Steps, No Guidance
            if (Transformer.ModelType == ModelType.Turbo)
            {
                return options with
                {
                    Steps = 4,
                    Shift = 1f,
                    Width = 512,
                    Height = 512,
                    GuidanceScale = 0,
                    Scheduler = SchedulerType.FlowMatchEulerDiscrete
                };
            }

            return options;
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        private bool _disposed;
        protected override void Dispose(bool disposing)
        {
            if (_disposed)
                return;
            if (disposing)
            {
                TextEncoder?.Dispose();
                Transformer?.Dispose();
                AutoEncoder?.Dispose();
            }
            _disposed = true;
        }
    }
}

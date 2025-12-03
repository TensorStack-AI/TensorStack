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
using TensorStack.TextGeneration.Tokenizers;
using QwenTextConfig = TensorStack.TextGeneration.Pipelines.Qwen.QwenConfig;
using QwenTextPipeline = TensorStack.TextGeneration.Pipelines.Qwen.QwenPipeline;

namespace TensorStack.StableDiffusion.Pipelines.Qwen
{
    public abstract class QwenBase : PipelineBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="QwenBase"/> class.
        /// </summary>
        /// <param name="transformer">The transformer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="autoEncoder">The automatic encoder.</param>
        /// <param name="logger">The logger.</param>
        public QwenBase(TransformerQwenModel transformer, QwenTextPipeline textEncoder, AutoEncoderModel autoEncoder, ILogger logger = default) : base(logger)
        {
            Transformer = transformer;
            TextEncoder = textEncoder;
            AutoEncoder = autoEncoder;
            Initialize();
            Logger?.LogInformation("[QwenPipeline] Name: {Name}", Name);
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="QwenBase"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public QwenBase(QwenConfig configuration, ILogger logger = default) : this(
            new TransformerQwenModel(configuration.Transformer),
            new QwenTextPipeline(new QwenTextConfig
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
        public override PipelineType PipelineType => PipelineType.Qwen;

        /// <summary>
        /// Gets the friendly name.
        /// </summary>
        public override string Name { get; init; } = nameof(PipelineType.Qwen);

        /// <summary>
        /// Gets the TextEncoder.
        /// </summary>
        public QwenTextPipeline TextEncoder { get; init; }

        /// <summary>
        /// Gets the transformer.
        /// </summary>
        public TransformerQwenModel Transformer { get; init; }

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
            // Qwen pipelines are lazy loaded on first run
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
            var cachedPrompt = GetPromptCache(options);
            if (cachedPrompt is not null)
                return cachedPrompt;

            // Conditional Prompt
            var promptEmbeds = await TextEncoder.GetLastHiddenState(new TextGeneration.Common.GenerateOptions
            {
                Seed = options.Seed,
                Prompt = options.Prompt,
                MinLength = 128,
                MaxLength = 128
            }, cancellationToken);

            // Unconditional prompt
            var negativePromptEmbeds = await TextEncoder.GetLastHiddenState(new TextGeneration.Common.GenerateOptions
            {
                Seed = options.Seed,
                Prompt = options.NegativePrompt,
                MinLength = 128,
                MaxLength = 128
            }, cancellationToken);

            return SetPromptCache(options, new PromptResult(promptEmbeds, default, negativePromptEmbeds, default));
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
            var decoderResult = await AutoEncoder.DecodeAsync(latents, disableShift: true, disableScale: true, cancellationToken: cancellationToken);
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
        private async Task<Tensor<float>> EncodeLatentsAsync(IPipelineOptions options, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[EncodeLatentsAsync] Begin AutoEncoder Encode");
            var cacheResult = GetEncoderCache(options);
            if (cacheResult is not null)
            {
                Logger.LogEnd(LogLevel.Debug, timestamp, "[EncodeLatentsAsync] AutoEncoder Encode Complete, Cached Result.");
                return cacheResult;
            }

            var inputTensor = options.InputImage.ResizeImage(options.Width, options.Height);
            var encoderResult = await AutoEncoder.EncodeAsync(inputTensor, cancellationToken: cancellationToken);
            if (options.IsLowMemoryEnabled || options.IsLowMemoryEncoderEnabled)
                await AutoEncoder.EncoderUnloadAsync();

            Logger.LogEnd(LogLevel.Debug, timestamp, "[EncodeLatentsAsync] AutoEncoder Encode Complete");
            return SetEncoderCache(options, encoderResult);
        }


        /// <summary>
        /// Run Transformer model inference
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="prompt">The prompt.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected async Task<Tensor<float>> RunInferenceAsync(IPipelineOptions options, IScheduler scheduler, PromptResult prompt, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var timestamp = Logger.LogBegin(LogLevel.Debug, "[RunInferenceAsync] Begin Transformer Inference");

            // Prompt
            var isGuidanceEnabled = IsGuidanceEnabled(options);
            var conditionalEmbeds = prompt.PromptEmbeds;
            var unconditionalEmbeds = prompt.NegativePromptEmbeds;

            // Latents
            var latents = await CreateLatentInputAsync(options, scheduler, cancellationToken);

            // Create ImgShapes
            var imgShapes = new Tensor<long>([1, 64, 64]); // TODO: H/W

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
                var conditional = await Transformer.RunAsync
                (
                    timestep,
                    latentInput,
                    conditionalEmbeds,
                    imgShapes,
                    cancellationToken: cancellationToken
                );

                // Guidance
                if (isGuidanceEnabled)
                {
                    var unconditional = await Transformer.RunAsync
                    (
                        timestep,
                        latentInput,
                        unconditionalEmbeds,
                        imgShapes,
                        cancellationToken: cancellationToken
                    );
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
            return UnpackLatents(latents, options.Width, options.Height);
        }



        /// <summary>
        /// Create latent input.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="scheduler">The scheduler.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        private async Task<Tensor<float>> CreateLatentInputAsync(IPipelineOptions options, IScheduler scheduler, CancellationToken cancellationToken = default)
        {
            if (options.HasInputImage)
            {
                var timestep = scheduler.GetStartTimestep();
                var encoderResult = await EncodeLatentsAsync(options, cancellationToken);
                var noiseTensor = scheduler.CreateRandomSample(encoderResult.Dimensions);
                return PackLatents(scheduler.ScaleNoise(timestep, encoderResult, noiseTensor));
            }

            var height = options.Height * 2 / AutoEncoder.LatentChannels;
            var width = options.Width * 2 / AutoEncoder.LatentChannels;
            return PackLatents(scheduler.CreateRandomSample([1, AutoEncoder.LatentChannels, height, width]));
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
        /// Packs the latents.
        /// </summary>
        /// <param name="latents">The latents.</param>
        /// <returns></returns>
        protected Tensor<float> PackLatents(Tensor<float> latents)
        {
            var height = latents.Dimensions[2] / 2;
            var width = latents.Dimensions[3] / 2;
            latents = latents.Reshape([1, AutoEncoder.LatentChannels, height, 2, width, 2]);
            latents = latents.Permute([0, 2, 4, 1, 3, 5]);
            latents = latents.Reshape([1, height * width, AutoEncoder.LatentChannels * 4]);
            return latents;
        }


        /// <summary>
        /// Unpacks the latents.
        /// </summary>
        /// <param name="latents">The latents.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <returns></returns>
        protected Tensor<float> UnpackLatents(Tensor<float> latents, int width, int height)
        {
            var channels = latents.Dimensions[2];
            height = height / AutoEncoder.LatentChannels;
            width = width / AutoEncoder.LatentChannels;
            latents = latents.Reshape([1, height, width, channels / 4, 2, 2]);
            latents = latents.Permute([0, 3, 1, 4, 2, 5]);
            latents = latents.Reshape([1, channels / (2 * 2), height * 2, width * 2]);
            return latents;
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
                Steps = 28,
                Shift = 1f,
                Width = 1024,
                Height = 1024,
                GuidanceScale = 3.5f,
                Scheduler = SchedulerType.FlowMatchEulerDiscrete
            };
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
